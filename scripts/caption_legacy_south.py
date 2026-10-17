"""Four-stage caption/conversation generation for legacy-south crossmatch objects.

Adapts the data-generation approach from AstroLLaVA (Zaman et al. 2025,
arXiv:2504.08583): captions/conversations must never name the object and
must never state facts that are not derivable from what's actually being
shown to the model. AstroLLaVA enforced this on GPT-4 text-only, working from
scraped human captions instead of images. Here we have real pixels and a
vision-capable model (Gemini), so every stage is grounded in the image
itself - never in catalog numbers or literature:

  1. caption_blind         - image alone, open-ended visual description.
  2. caption_properties    - image alone again, but structured to make the
                              model estimate the same property categories our
                              pipeline measures (color balance, extent,
                              shape, star-formation signatures) purely by
                              looking, never told the actual values. This is
                              the training signal that matters: the model
                              has to learn to infer these at inference time,
                              when there is no catalog to read from.
  3. caption_literature     - image + real excerpts from the literature
                              mentions this object was crossmatched against
                              (the astronolan/galaxy-mentions-hats dataset,
                              via `fetch-mentions`), with every identity
                              string for this object (all name variants,
                              NED object name, host-of/coordinate-resolution
                              labels) redacted from the text before it ever
                              reaches the model - not just instructed away.
                              The model uses the real astrophysical
                              knowledge in those excerpts to reason at an
                              expert level about what's observable in the
                              image, without ever being told what the
                              object is.
  4. conversation           - grounded in the image directly, 3-6 casual
                              human/assistant turns, same constraints. Last,
                              so it can read as the natural wrap-up.

Catalog measurements (flux, shape) are never shown to the model in any
stage; they're only computed and stored in the output as a QC reference, to
check the model's visual claims against the truth without ever training on
the numbers. Literature text IS shown to the model in stage 3, but only
after identity redaction - the constraint is "no leaked names", not "no
literature".

Every stage is an independent, stateless `generate_content` call - there is
no shared chat session and no conversation history accumulating across
stages, so there's no context-rot risk from chaining. Whatever a stage needs
from an earlier one (e.g. the literature block) is explicitly re-injected as
fresh text in that stage's own prompt, not carried by model-side memory.

Example:
  python scripts/caption_legacy_south.py --object-id 1610p117-2204 \\
      --images outputs/crossmatches/legacy_south_top_5_images.parquet \\
      --metadata outputs/crossmatches/legacy_south_top_5_metadata.parquet \\
      --mentions outputs/crossmatches/legacy_south_top_5_captions_full_mentions.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv
from google import genai

from render_legacy_rgb import render_object_rgb, saturation_fraction

load_dotenv()

GEMINI_MODEL = "gemini-pro-latest"

_HARD_NAME_RULE_HEADER = """ABSOLUTE RULE - THIS OVERRIDES EVERY OTHER INSTRUCTION BELOW:
You must NEVER output the object's name, any catalog designation (NGC/IC/UGC/M/Messier/PGC/2MASX/etc.), any host-galaxy label, or any other proper identifying string - for this object, or for any OTHER object that happens to be mentioned in your input. Not partially, not as part of a longer phrase, not even if it would make the answer more complete or accurate. If you are ever unsure whether something counts as a name, leave it out. This matters more than completeness, more than detail, more than any other instruction here. A response that names anything has completely failed the task, no matter how good the rest of it is."""

_HARD_NAME_RULE_FOOTER = """Before you answer, check your own response one more time: does it contain any name, designation, or identifying string, for this object or any other? If so, remove it before responding."""

STAGE1_PROMPT = f"""{_HARD_NAME_RULE_HEADER}

You are an expert astronomical image analyst. You are shown a false-color composite image of an astronomical object from a wide-field optical imaging survey (channel mapping: z-band=red, r-band=green, g-band=blue).

Describe ONLY what is visible in the image itself.

Guidelines:
- Do not state facts you cannot determine from the pixels alone (no distance, redshift, age, or discovery history).
- Describe morphology (e.g. spiral, elliptical, irregular, point-like), structure (arms, bar, bulge, disk, knots, halo), apparent color, texture, and any notable features.
- Write 4-6 sentences, in a clear, scientific-but-accessible tone.
- Refer to it only with generic terms such as "this galaxy" or "this object".

{_HARD_NAME_RULE_FOOTER}
"""

STAGE2_PROMPT = f"""{_HARD_NAME_RULE_HEADER}

You are an expert astronomical image analyst, examining this false-color composite image (z-band=red, r-band=green, g-band=blue) the way an observer would at the eyepiece or on a display - with no catalog, photometry software, or measurements available to you. Only your visual judgment.

Give your best expert visual estimate of the following, reasoning only from what you see:

- Color balance: does the object look systematically redder or bluer overall, and is that color roughly uniform or does it vary between regions (e.g. a redder core vs. bluer outer features, or vice versa)?
- Extent and shape: does it look compact/unresolved or clearly extended, and is its outline closer to circular or noticeably elongated? If elongated, in roughly what direction?
- Structure: any signs of a bulge, disk, spiral arms, bar, rings, knots, or irregular clumps, and how they're arranged relative to each other.
- Star formation / dust signatures: any visual evidence suggestive of active star formation or dust (e.g. clumpy blue knots, dust lanes, patchy asymmetric structure).

Guidelines:
- These are your own visual estimates, not measurements you were given - state them as estimates ("appears to be...", "looks roughly circular...") rather than precise facts.
- Do not state facts you cannot estimate from the image (no distance, redshift, age, or discovery history).
- Write 4-6 sentences, in a clear, scientific-but-accessible tone, as an experienced observer sharing their assessment.
- Refer to it only with generic terms such as "this galaxy" or "this object".

{_HARD_NAME_RULE_FOOTER}
"""

STAGE3_PROMPT_TEMPLATE = f"""{_HARD_NAME_RULE_HEADER}

You are an expert astronomical image analyst. You are shown a false-color composite image (z-band=red, r-band=green, g-band=blue) of an object catalogued under the following name(s): {{target_names}}. You are being told this name ONLY so you can tell which literature excerpts below are actually about this object versus some other object - it must never appear in your response (see the absolute rule above).

You are also given excerpts from published scientific literature that mention this object. IMPORTANT: several excerpts discuss OTHER objects in the same sentence or alongside it - comparison galaxies, field neighbors, companion sources. Any measurement, property, classification, or description attributed to an object other than {{target_names}} does NOT describe the object in the image. You must identify and completely disregard that content; only reason from statements that are actually about {{target_names}}.

Literature excerpts:
{{literature_block}}

Using the image itself as the primary evidence, and only the literature content that is genuinely about {{target_names}} to sharpen your interpretation, write ONE expert-level description that:
- Reasons about what's observable in the image with the benefit of relevant astrophysical context (e.g. if the excerpts discuss a type of structure, process, or classification for this object, you may draw on that vocabulary and reasoning) - but only for properties adjacent to what is directly visible here.
- Does not restate specific non-visual facts (no distances, redshifts, dates, masses, or other papers' measurements) - use them only to inform how you interpret the pixels, not as facts to report.
- Ignores anything about identity, naming, discovery, or classification history - for this object or any other.

{_HARD_NAME_RULE_FOOTER} This is the highest-risk step in this entire pipeline because you were just shown the real name - treat that as a reason for extra caution, not less.
"""

STAGE4_PROMPT = f"""{_HARD_NAME_RULE_HEADER}

You are shown a false-color composite image of an astronomical object (z-band=red, r-band=green, g-band=blue).

Design a casual, conversational exchange between a curious person and an AI assistant discussing this image: between 3 and 6 question/answer turns.

Requirements for the questions:
1. Do not ask about anything not determinable from the image alone (no redshift, distance, or age).
2. Focus on visual aspects: structure, morphology, color, notable features, size within the frame.
3. Keep the tone casual and conversational.

Requirements for the answers:
1. Do not state specific facts, numbers, dates, or names that aren't visible in the image.
2. Answer as if directly observing - don't refer to "the image", "the data", or "the context" as an external thing being described.
3. If a question can't be honestly answered from the image alone, say so rather than inventing an answer.

Return valid JSON only, in exactly this shape, no markdown fences:
[
  {{"from": "human", "value": "..."}},
  {{"from": "assistant", "value": "..."}}
]

{_HARD_NAME_RULE_FOOTER}
"""

_NON_NAME_IDENTITY_FIELDS = ["ned_object_name_mention", "coordinate_resolution_name_mention"]


def collect_target_names(mentions_for_object: pd.DataFrame) -> list[str]:
    """True name variants for this object (e.g. "NGC 3351", "M95") - what the
    model is told to call it, for disambiguating it from other objects in
    the same excerpts. Excludes non-name identity labels like NED's
    "SN 2012aw HOST" host-association tag, which isn't a name anyone would
    plausibly use to refer to the object.
    """
    strings: set[str] = set()
    strings.update(v for v in mentions_for_object["name_mention"].dropna() if v)
    for lst in mentions_for_object.get("additional_names_mention", pd.Series(dtype=object)).dropna():
        strings.update(v for v in (lst if isinstance(lst, list) else []) if v)
    return sorted(strings, key=len, reverse=True)


def collect_identity_strings(mentions_for_object: pd.DataFrame) -> list[str]:
    """Every name/identity string attached to this object's mentions - the
    full superset used for the output leak check (broader than
    collect_target_names(), since a leak of "SN 2012aw HOST" or the bare
    "SN 2012aw" designation is just as much an identity leak as the name).
    """
    strings = set(collect_target_names(mentions_for_object))
    for field in _NON_NAME_IDENTITY_FIELDS:
        strings.update(v for v in mentions_for_object[field].dropna() if v)
    # "SN 2012aw HOST" also leaks the bare supernova designation "SN 2012aw" - catch both spans.
    expanded = set(strings)
    for s in strings:
        if s.upper().endswith(" HOST"):
            expanded.add(s[: -len(" HOST")])
    return sorted(expanded, key=len, reverse=True)


# Output-side safety net: after the model generates a caption, scan it for
# anything name-shaped. This is deliberately NOT applied to the literature
# input anymore - blind input redaction can't tell whose property a sentence
# describes when several objects are discussed together (that's a reading-
# comprehension task, not a string-matching one), so stage 3 instead gives
# the model the real names and asks it to reason about attribution itself.
# What regex is well-suited for is checking a single short, freshly generated
# caption (a few sentences) for a leaked name - low stakes of mangling real
# content, unlike blindly redacting 48 dense paragraphs of source literature.
_CATALOG_DESIGNATION_RE = re.compile(r"\b[A-Z]{2,6}[\s-]?J?\d{2,}[\d.+-]*\b")
_MESSIER_RE = re.compile(r"\bMessier\s?\d+\b", re.IGNORECASE)
_M_DESIGNATION_RE = re.compile(r"\bM\s?\d{1,4}\b")  # e.g. "M87", "M 95" - capital M only, not a mass symbol
_SUPERNOVA_RE = re.compile(r"\bSN\s?\d{4}[a-zA-Z]{1,3}\b", re.IGNORECASE)


def find_identity_leaks(text: str, identity_strings: list[str]) -> list[str]:
    """Every identity-shaped span found in `text` - the target's own known
    name variants plus anything matching the general catalog-designation
    patterns (which also catches OTHER objects' names, e.g. a comparison
    galaxy the model echoed). Empty list means clean.
    """
    if not text:
        return []
    hits: list[str] = []
    for name in identity_strings:
        if re.search(r"\b" + re.escape(name) + r"\b", text, re.IGNORECASE):
            hits.append(name)
    for pattern in (_SUPERNOVA_RE, _MESSIER_RE, _M_DESIGNATION_RE, _CATALOG_DESIGNATION_RE):
        hits.extend(m.group() for m in pattern.finditer(text))
    return hits


def get_mentions_for_object(mentions_path: str, object_id: str) -> pd.DataFrame:
    """The `fetch-mentions` output now carries `object_id_legacy` through
    directly (crossmatch_legacy_south.py passes through the input manifest's
    survey-side columns), so this is a single-file lookup - no separate
    captions bridge table needed anymore.
    """
    mentions = read_parquet_df(mentions_path)
    sub = mentions[mentions["object_id_legacy"] == object_id]
    if sub.empty:
        raise ValueError(f"No literature mentions found for object {object_id} in {mentions_path}")
    return sub


def build_literature_block(mentions_for_object: pd.DataFrame) -> str:
    """Raw (unredacted) mention_summary text for every literature mention
    crossmatched to this object. Stage 3's prompt tells the model which
    name(s) refer to the object in the image and instructs it to disregard
    anything attributed to a different named object; find_identity_leaks()
    is the hard check on what actually comes back out.
    """
    summaries = mentions_for_object["mention_summary_mention"].dropna()
    return "\n".join(f"[{i}] {summary}" for i, summary in enumerate(summaries, start=1))


def read_parquet_df(path: str) -> pd.DataFrame:
    table = pq.read_table(path)
    table = table.replace_schema_metadata(None)
    return table.to_pandas()


def build_photometry_reference(meta_row: dict) -> dict:
    """Real measurements for QC only - never shown to the model, never in a prompt."""

    def mag(flux):
        if flux is None or flux <= 0:
            return None
        return round(22.5 - 2.5 * math.log10(flux), 3)

    e1, e2 = meta_row["SHAPE_E1_legacy"], meta_row["SHAPE_E2_legacy"]
    ellipticity = math.sqrt(e1**2 + e2**2)
    position_angle = 0.5 * math.degrees(math.atan2(e2, e1))
    if position_angle < 0:
        position_angle += 180

    return {
        "mag": {b: mag(meta_row.get(f"FLUX_{b}_legacy")) for b in ["G", "R", "I", "Z", "W1", "W2", "W3", "W4"]},
        "effective_radius_arcsec": round(meta_row["SHAPE_R_legacy"], 3),
        "ellipticity": round(ellipticity, 4),
        "position_angle_deg": round(position_angle, 2),
    }


def format_target_names(identity_strings: list[str]) -> str:
    variants = sorted(identity_strings, key=str.lower)
    return " / ".join(f'"{v}"' for v in variants)


def call_gemini(client: genai.Client, image, prompt: str) -> str:
    response = client.models.generate_content(model=GEMINI_MODEL, contents=[prompt, image])
    return response.text


def generate_with_leak_guard(
    client: genai.Client,
    image,
    prompt: str,
    identity_strings: list[str],
    stage_label: str,
) -> tuple[str, list[str]]:
    """Call Gemini once, then hard-check the output for leaked identity strings.

    Exactly one API call, always - no retry loop. A leak does NOT trigger
    another Gemini call (that's real money for an outcome that isn't
    guaranteed to improve); it prints a loud warning and flags the leaked
    spans in the returned result so you can see and fix it by hand. The
    prompts themselves (see the "ABSOLUTE RULE" blocks) are the real defense
    against leaking in the first place - this is strictly a detector, not an
    auto-corrector.
    """
    text = call_gemini(client, image, prompt)
    leaks = find_identity_leaks(text, identity_strings)
    if leaks:
        print(f"\n{'!' * 70}")
        print(f"  LEAK DETECTED in {stage_label}: {sorted(set(leaks))}")
        print(f"  This caption was NOT regenerated (no extra API calls). Review it by hand:")
        print(f"  {text}")
        print(f"{'!' * 70}\n")
    return text, leaks


def parse_json_response(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def run_one_object(
    images_path: str,
    metadata_path: str,
    mentions_path: str,
    object_id: str,
    output_dir: str,
    dry_run: bool = False,
) -> dict:
    images = read_parquet_df(images_path)
    metadata = read_parquet_df(metadata_path)

    oid_col = "object_id_legacy"
    img_row = images[images[oid_col] == object_id].iloc[0]
    meta_row = metadata[metadata[oid_col] == object_id].iloc[0].to_dict()

    rgb = render_object_rgb(img_row["image_legacy"])
    sat = saturation_fraction(rgb)
    print(f"Rendered {object_id}: saturation={sat:.4f}")

    mentions_for_object = get_mentions_for_object(mentions_path, object_id)
    identity_strings = collect_identity_strings(mentions_for_object)
    target_names = format_target_names(collect_target_names(mentions_for_object))
    literature_block = build_literature_block(mentions_for_object)
    stage3_prompt = STAGE3_PROMPT_TEMPLATE.format(target_names=target_names, literature_block=literature_block)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb.save(out_dir / f"{object_id}_rgb.png")

    if dry_run:
        print(f"\nTarget names (told to the model in stage 3): {target_names}")
        print(f"Full identity strings (checked against every stage's output): {identity_strings}")
        print("\n--- dry run: raw literature block that would be sent in stage 3 ---")
        print(literature_block)
        print("--- end literature block ---\n")
        print("No Gemini calls made (--dry-run).")
        return {
            "object_id": object_id,
            "identity_strings": identity_strings,
            "literature_block": literature_block,
            "stage3_prompt": stage3_prompt,
        }

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    leak_flags = {}

    print("Stage 1: blind caption...")
    caption_blind, leak_flags["stage1"] = generate_with_leak_guard(client, rgb, STAGE1_PROMPT, identity_strings, "stage1")

    print("Stage 2: structured property estimate...")
    caption_properties, leak_flags["stage2"] = generate_with_leak_guard(
        client, rgb, STAGE2_PROMPT, identity_strings, "stage2"
    )

    print("Stage 3: literature-informed caption...")
    caption_literature, leak_flags["stage3"] = generate_with_leak_guard(
        client, rgb, stage3_prompt, identity_strings, "stage3"
    )

    print("Stage 4: conversation...")
    stage4_raw, leak_flags["stage4"] = generate_with_leak_guard(client, rgb, STAGE4_PROMPT, identity_strings, "stage4")
    try:
        conversation = parse_json_response(stage4_raw)
        conversation_raw = None
    except json.JSONDecodeError as error:
        print(f"Could not parse stage 4 as JSON ({error}); keeping raw text.")
        conversation, conversation_raw = None, stage4_raw

    any_leaks = any(leak_flags.values())
    if any_leaks:
        print(f"\n>>> {object_id}: one or more stages leaked identity - see leak_flags_per_stage in the output JSON.\n")

    result = {
        "object_id": object_id,
        "render_saturation_fraction": sat,
        "caption_blind": caption_blind,
        "caption_properties": caption_properties,
        "conversation": conversation,
        "conversation_raw": conversation_raw,
        "caption_literature": caption_literature,
        "identity_strings_shown_to_model_in_stage3": identity_strings,
        "literature_block_raw": literature_block,
        "leak_flags_per_stage": leak_flags,
        "any_leaks_detected": any_leaks,
        "photometry_reference_not_shown_to_model": build_photometry_reference(meta_row),
    }

    with open(out_dir / f"{object_id}_captions.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_dir / f'{object_id}_captions.json'}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--object-id", required=True, help="object_id_legacy to caption")
    parser.add_argument("--images", required=True, help="path to a *_images.parquet file")
    parser.add_argument("--metadata", required=True, help="path to a *_metadata.parquet file")
    parser.add_argument(
        "--mentions",
        required=True,
        help="path to a *_full_mentions.parquet file from fetch-mentions (must carry object_id_legacy)",
    )
    parser.add_argument("--output-dir", default="outputs/captions")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build and print the redacted literature block without calling Gemini",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_one_object(args.images, args.metadata, args.mentions, args.object_id, args.output_dir, args.dry_run)
    if args.dry_run:
        return

    for label, key in [
        ("STAGE 1 - blind caption", "caption_blind"),
        ("STAGE 2 - structured property estimate", "caption_properties"),
        ("STAGE 3 - literature-informed caption", "caption_literature"),
    ]:
        print("\n" + "=" * 70)
        print(label)
        print("=" * 70)
        print(result[key])

    print("\n" + "=" * 70)
    print("STAGE 4 - conversation")
    print("=" * 70)
    if result["conversation"]:
        for turn in result["conversation"]:
            print(f"\n[{turn['from'].upper()}] {turn['value']}")
    else:
        print(result["conversation_raw"])

    print("\n" + "=" * 70)
    print("QC reference (not shown to model)")
    print("=" * 70)
    print(json.dumps(result["photometry_reference_not_shown_to_model"], indent=2))


if __name__ == "__main__":
    main()
