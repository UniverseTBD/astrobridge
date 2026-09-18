#!/usr/bin/env python
"""Ask a free-form question about a single, brand-new object — not part of the manifest/cache
pipeline. Runs the AION encoders live instead of reading from outputs/cache/.

Image input: a .npy file shaped (n_bands, H, W), band order matching configs/modalities.yaml's
    modalities.image.encoder.kwargs.bands.
Spectrum input: a .npz file with arrays `flux`, `wavelength` (both (L,)), optionally `ivar`,
    `mask` (default to fully-trusted/fully-unmasked if omitted — see aion_spectrum.py), plus
    `--spectrum-survey desi|sdss` (required, no safe default — see aion_spectrum.py's docstring
    for why AION needs to know real survey origin).
Light-curve input: a .npz file with arrays `mjd`, `flux`, `flux_err`, `band_id` (all (L,)), and
    optionally `use` (bool (L,), defaults to all-True). Flux must be in SNANA FLUXCAL at zero
    point 27.5 and `band_id` must use ATCAT's convention (g=1, r=2) — the same units the cached
    embeddings were built from. It is run through the identical detection-window trim, seeded
    downsample and pad-to-243 that scripts/02_cache_embeddings.py applies.

At least one of --image-npy / --spectrum-npz / --lightcurve-npz must be given; they may be given
together to condition the answer on several modalities at once. --question is free text — the
frozen base LLM's own
instruction-following is what's being relied on here, not something LoRA/the fusion stack were
specifically trained to do (they only ever saw one fixed captioning instruction during
training) — see inference.py's generate_caption docstring.

Usage:
    python scripts/07_infer.py --checkpoint-dir outputs/checkpoints/stage2/best \\
        --lora-dir outputs/checkpoints/stage2/best/lora --image-npy my_cutout.npy \\
        --question "What kind of object is this and why do you think so?"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from captioner.data.transients_dataset import prepare_lightcurve_arrays
from captioner.data.spectra_dataset import trimmed_spectrum_arrays
from captioner.inference import generate_caption, load_inference_model
from captioner.utils.config import load_config, remaining_argv
from captioner.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, help="e.g. outputs/checkpoints/stage2/best")
    parser.add_argument("--lora-dir", default=None, help="e.g. outputs/checkpoints/stage2/best/lora")
    parser.add_argument("--question", required=True, help="free-form question/instruction, e.g. "
                         "'What kind of object is this?'")
    parser.add_argument("--image-npy", default=None, help="path to a (n_bands, H, W) .npy file")
    parser.add_argument("--spectrum-npz", default=None, help="path to a .npz with flux/wavelength/...")
    parser.add_argument("--spectrum-survey", default=None, choices=["desi", "sdss"])
    parser.add_argument("--lightcurve-npz", default=None, help="path to a .npz with mjd/flux/flux_err/band_id")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args(remaining_argv())

    if args.image_npy is None and args.spectrum_npz is None and args.lightcurve_npz is None:
        parser.error("At least one of --image-npy, --spectrum-npz or --lightcurve-npz is required.")
    if args.spectrum_npz is not None and args.spectrum_survey is None:
        parser.error("--spectrum-survey is required when --spectrum-npz is given.")

    cfg = load_config("base", "data", "modalities", "model", "stage2")

    logger.info("Loading model (this can take a few minutes with no output — that's normal)...")
    model, tokenizer, encoders = load_inference_model(cfg, args.checkpoint_dir, args.lora_dir, args.device)
    logger.info("Model loaded.")

    raw_inputs: dict = {}
    if args.image_npy is not None:
        pixel_values = torch.from_numpy(np.load(args.image_npy)).unsqueeze(0)  # (1, n_bands, H, W)
        raw_inputs["image"] = {"pixel_values": pixel_values}
    if args.spectrum_npz is not None:
        npz = np.load(args.spectrum_npz)
        # Trim `wavelength <= 0` and sort ascending before AION ever sees it. AstroBridge-Data
        # pads its wavelength grids with a -1 sentinel, and AION's codec interpolates with
        # torch.searchsorted, which silently returns garbage on an unsorted grid — that is what
        # made every v5/v6 spectrum embedding near-identical. aion_spectrum.py hard-errors on a
        # non-monotonic grid now, so without this a perfectly ordinary SDSS .npz would be
        # rejected rather than quietly mis-encoded.
        flux, wavelength, ivar, mask = trimmed_spectrum_arrays(
            {
                "flux": npz["flux"],
                "lambda": npz["wavelength"],
                "ivar": npz["ivar"] if "ivar" in npz else np.ones_like(npz["flux"]),
                "mask": npz["mask"] if "mask" in npz else np.zeros_like(npz["flux"], dtype=bool),
            }
        )
        spectrum_batch: dict = {
            "flux": torch.from_numpy(flux).unsqueeze(0).float(),
            "wavelength": torch.from_numpy(wavelength).unsqueeze(0).float(),
            "ivar": torch.from_numpy(ivar).unsqueeze(0).float(),
            "mask": torch.from_numpy(mask).unsqueeze(0).bool(),
            "survey": [args.spectrum_survey],
        }
        raw_inputs["spectra"] = spectrum_batch

    if args.lightcurve_npz is not None:
        npz = np.load(args.lightcurve_npz)
        required = ("mjd", "flux", "flux_err", "band_id")
        absent = [k for k in required if k not in npz]
        if absent:
            parser.error(
                f"{args.lightcurve_npz} is missing array(s) {absent}; required: {list(required)} "
                "(plus optional `use`)."
            )
        use = npz["use"] if "use" in npz else np.ones(len(npz["mjd"]), dtype=bool)
        lc_modality = cfg.modalities.lightcurve
        lc_kwargs = lc_modality.encoder.get("kwargs", {})
        arrays, info = prepare_lightcurve_arrays(
            npz["mjd"], npz["flux"], npz["flux_err"], npz["band_id"], use,
            object_id=Path(args.lightcurve_npz).stem,
            seq_len=int(lc_modality.max_tokens),
            detection_window_days=float(lc_kwargs.get("detection_window_days", 30.0)),
            detection_snr=float(lc_kwargs.get("detection_snr", 5.0)),
            seed=int(lc_kwargs.get("subsample_seed", 0)),
        )
        logger.info(
            f"Light curve: {info['n_accepted']} accepted points, {info['n_in_window']} inside the "
            f"detection window, {info['n_selected']} encoded."
        )
        raw_inputs["lightcurve"] = {k: torch.from_numpy(v[None, :]) for k, v in arrays.items()}

    out_dims = {n: int(c.out_dim) for n, c in cfg.modalities.items()}
    max_tokens = {n: int(c.max_tokens) for n, c in cfg.modalities.items()}

    answer = generate_caption(
        model, tokenizer, encoders, out_dims, max_tokens, cfg.prompt, args.device,
        raw_inputs, max_new_tokens=args.max_new_tokens, question=args.question,
    )
    logger.info(f"Answer: {answer}")


if __name__ == "__main__":
    main()
