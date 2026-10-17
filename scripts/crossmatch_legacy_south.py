"""Create a lightweight literature × survey match manifest.

Examples:
  python scripts/crossmatch_legacy_south.py match --survey manga
  python scripts/crossmatch_legacy_south.py top-k \
      outputs/crossmatches/legacy_north_manifest.parquet --survey legacy-north --k 100
  python scripts/crossmatch_legacy_south.py fetch-images \
      outputs/crossmatches/legacy_north_top_100_objects.parquet --survey legacy-north

`match` intentionally opens no image columns. The resulting
manifest is the input to a later, targeted heavy-modality retrieval step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lsdb
import pandas as pd

MENTIONS = "hf://datasets/astronolan/galaxy-mentions-hats"
SURVEYS = {
    "legacy-south": {
        "uri": "hf://datasets/hugging-science/mmu_legacysurvey_dr10_south_21",
        "suffix": "legacy",
        "id_column": "object_id",
        "ra_column": "ra",
        "dec_column": "dec",
        "light_columns": [],
        "image_columns": ["image", "rgb"],
        "payload_columns": ["image", "blobmodel", "rgb", "object_mask", "catalog"],
    },
    "legacy-north": {
        "uri": "hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north",
        "suffix": "legacy",
        "id_column": "object_id",
        "ra_column": "ra",
        "dec_column": "dec",
        "light_columns": ["z_spec"],
        "image_columns": ["image", "z_spec"],
        "payload_columns": ["image"],
    },
    "manga": {
        "uri": "hf://datasets/hugging-science/mmu_manga",
        "suffix": "manga",
        "id_column": "object_id",
        "ra_column": "ra",
        "dec_column": "dec",
        "light_columns": ["z"],
        "image_columns": ["images", "z"],
        "payload_columns": ["images", "spaxels", "maps"],
    },
    "hsc": {
        "uri": "hf://datasets/UniverseTBD/mmu_hsc_pdr3_dud_22.5",
        "suffix": "hsc",
        "id_column": "object_id",
        "ra_column": "ra",
        "dec_column": "dec",
        "light_columns": [],
        "image_columns": ["image"],
        "payload_columns": ["image"],
    },
    "galaxy-zoo": {
        "uri": "hf://datasets/UniverseTBD/mmu_gz10",
        "suffix": "gz10",
        "id_column": "object_id",
        "ra_column": "ra",
        "dec_column": "dec",
        "light_columns": ["redshift", "gz10_label"],
        "image_columns": ["rgb_image", "redshift", "gz10_label", "rgb_pixel_scale"],
        "payload_columns": ["rgb_image"],
    },
    "des-dr2": {
        "uri": "https://linea.data.lsdb.io/hats/des/des_dr2",
        "suffix": "des",
        "id_column": "COADD_OBJECT_ID",
        "ra_column": "RA",
        "dec_column": "DEC",
        "light_columns": [],
        "image_columns": [],
        "payload_columns": [],
    },
}
MENTION_COLUMNS = [
    "mention_id",
    "ra",
    "dec",
    "name",
    "additional_names",
    "coordinate_resolution_name",
    "mention_summary",
    "uat_terms",
    "discussion_extent",
    "arxiv_id",
    "resolver_api_or_agent",
    "resolver_method_used",
    "resolver_agent_confidence",
    "ned_object_name",
    "ned_physical_type",
    "ned_position_refcode",
    "ned_url",
    "evidence_quote_count",
    "evidence_quotes",
    "wiki_entity_id",
]


def require_columns(catalog: object, requested: list[str], label: str) -> None:
    """Raise a useful error before an expensive computation starts."""
    available = set(catalog.columns)  # type: ignore[attr-defined]
    missing = [column for column in requested if column not in available]
    if missing:
        raise ValueError(
            f"{label} is missing columns {missing}. Available columns: "
            f"{sorted(available)}"
        )


def catalog_uri(args: argparse.Namespace) -> str:
    return args.catalog or SURVEYS[args.survey]["uri"]


def selected_object_column(survey: str) -> str:
    return f"object_id_{SURVEYS[survey]['suffix']}"


def catalog_key_columns(survey: str) -> list[str]:
    profile = SURVEYS[survey]
    return [profile["id_column"], profile["ra_column"], profile["dec_column"]]


def normalize_catalog_columns(frame: pd.DataFrame, survey: str) -> pd.DataFrame:
    """Use stable output names even when a source uses uppercase column names."""
    profile = SURVEYS[survey]
    suffix = profile["suffix"]
    return frame.rename(
        columns={
            f"{profile['id_column']}_{suffix}": f"object_id_{suffix}",
            f"{profile['ra_column']}_{suffix}": f"ra_{suffix}",
            f"{profile['dec_column']}_{suffix}": f"dec_{suffix}",
        }
    )


def parse_selection_count(value: str) -> int | None:
    """Parse a positive top-K count, or all for the full intersection."""
    if value.lower() == "all":
        return None
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--k must be a positive integer or 'all'") from error
    if count < 1:
        raise argparse.ArgumentTypeError("--k must be a positive integer or 'all'")
    return count


def command_match(args: argparse.Namespace) -> None:
    mentions = lsdb.open_catalog(args.mentions, columns=MENTION_COLUMNS)
    catalog_columns = [*catalog_key_columns(args.survey), *SURVEYS[args.survey]["light_columns"]]
    legacy = lsdb.open_catalog(catalog_uri(args), columns=catalog_columns)
    require_columns(mentions, MENTION_COLUMNS, "mentions catalog")
    require_columns(legacy, catalog_columns, f"{args.survey} catalog")

    matches = mentions.crossmatch(
        legacy,
        radius_arcsec=args.radius_arcsec,
        n_neighbors=1,
        suffixes=("_mention", f"_{SURVEYS[args.survey]['suffix']}"),
        suffix_method="all_columns",
    )
    # LSDB returns a NestedFrame.  Convert its lightweight result to a regular
    # pandas DataFrame before writing; NestedFrame.to_parquet has no `index=`
    # parameter, unlike pandas.DataFrame.to_parquet.
    manifest = normalize_catalog_columns(pd.DataFrame(matches.compute()), args.survey)

    output = Path(
        args.output or f"outputs/crossmatches/{args.survey.replace('-', '_')}_manifest.parquet"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output, index=False)
    print(f"Wrote {len(manifest):,} matched mentions to {output}")
    print(f"Unique {args.survey} objects: {manifest[selected_object_column(args.survey)].nunique():,}")


def command_top_k(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    manifest = pd.read_parquet(manifest_path)
    object_id = selected_object_column(args.survey)
    suffix = SURVEYS[args.survey]["suffix"]
    mention_id = "mention_id_mention"
    required = {object_id, mention_id, f"ra_{suffix}", f"dec_{suffix}"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"{manifest_path} is not a {args.survey} manifest; missing {missing}")

    counts = (
        manifest.groupby(object_id)[mention_id]
        .nunique()
        .rename("caption_count")
        .sort_values(ascending=False)
    )
    selected_ids = counts.index if args.k is None else counts.head(args.k).index
    captions = manifest[manifest[object_id].isin(selected_ids)].copy()
    objects = (
        captions.sort_values("_dist_arcsec")
        .drop_duplicates(object_id)
        [[object_id, f"ra_{suffix}", f"dec_{suffix}"]]
        .merge(counts.rename_axis(object_id).reset_index(), on=object_id)
        .sort_values(["caption_count", object_id], ascending=[False, True])
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = "all" if args.k is None else f"top_{args.k}"
    stem = f"{args.survey.replace('-', '_')}_{selection}"
    captions_path = output_dir / f"{stem}_captions.parquet"
    objects_path = output_dir / f"{stem}_objects.parquet"
    captions.to_parquet(captions_path, index=False)
    objects.to_parquet(objects_path, index=False)
    print(f"Wrote {len(objects):,} objects to {objects_path}")
    print(f"Wrote {len(captions):,} matched captions to {captions_path}")


def command_fetch_images(args: argparse.Namespace) -> None:
    """Retrieve heavy columns only for objects selected by ``top-k``."""
    objects_path = Path(args.objects)
    objects = pd.read_parquet(objects_path)
    suffix = SURVEYS[args.survey]["suffix"]
    object_id = selected_object_column(args.survey)
    required = {object_id, f"ra_{suffix}", f"dec_{suffix}"}
    missing = sorted(required.difference(objects.columns))
    if missing:
        raise ValueError(f"{objects_path} is not a {args.survey} object subset; missing {missing}")

    targets_df = (
        objects[[object_id, f"ra_{suffix}", f"dec_{suffix}"]]
        .drop_duplicates(object_id)
        .rename(
            columns={
                object_id: "target_object_id",
                f"ra_{suffix}": "ra",
                f"dec_{suffix}": "dec",
            }
        )
    )
    targets = lsdb.from_dataframe(targets_df, ra_column="ra", dec_column="dec")
    image_columns = args.columns or SURVEYS[args.survey]["image_columns"]
    if not image_columns:
        raise ValueError(f"{args.survey} has no image payload; use match/top-k only.")
    columns = [*catalog_key_columns(args.survey), *image_columns]
    legacy = lsdb.open_catalog(catalog_uri(args), columns=columns)
    require_columns(legacy, columns, f"{args.survey} catalog")

    retrieved = normalize_catalog_columns(pd.DataFrame(
        targets.crossmatch(
            legacy,
            radius_arcsec=args.radius_arcsec,
            n_neighbors=1,
            suffixes=("_target", f"_{suffix}"),
            suffix_method="all_columns",
        ).compute()
    ), args.survey)
    if len(retrieved) != len(targets_df):
        raise RuntimeError(
            f"Retrieved {len(retrieved)} of {len(targets_df)} requested objects; "
            "increase --radius-arcsec only if the selected coordinates are not "
            "copied directly from the match manifest."
        )
    if not (
        retrieved["target_object_id_target"].astype(str)
        == retrieved[object_id].astype(str)
    ).all():
        raise RuntimeError("Spatial retrieval returned an object ID different from its target ID")

    default_name = objects_path.name.replace("_objects.parquet", "_images.parquet")
    output = Path(args.output or objects_path.with_name(default_name))
    output.parent.mkdir(parents=True, exist_ok=True)
    retrieved.to_parquet(output, index=False)
    print(f"Wrote {len(retrieved):,} image rows to {output}")



def command_fetch_metadata(args: argparse.Namespace) -> None:
    """Retrieve all non-image survey columns for a selected object subset."""
    objects_path = Path(args.objects)
    objects = pd.read_parquet(objects_path)
    suffix = SURVEYS[args.survey]["suffix"]
    object_id = selected_object_column(args.survey)
    required = {object_id, f"ra_{suffix}", f"dec_{suffix}"}
    missing = sorted(required.difference(objects.columns))
    if missing:
        raise ValueError(f"{objects_path} is not a {args.survey} object subset; missing {missing}")

    targets_df = (
        objects[[object_id, f"ra_{suffix}", f"dec_{suffix}"]]
        .drop_duplicates(object_id)
        .rename(
            columns={
                object_id: "target_object_id",
                f"ra_{suffix}": "ra",
                f"dec_{suffix}": "dec",
            }
        )
    )
    targets = lsdb.from_dataframe(targets_df, ra_column="ra", dec_column="dec")

    profile = SURVEYS[args.survey]
    schema = lsdb.open_catalog(catalog_uri(args))
    metadata_columns = [
        column for column in schema.columns if column not in profile["payload_columns"]
    ]
    catalog = lsdb.open_catalog(catalog_uri(args), columns=metadata_columns)
    require_columns(catalog, metadata_columns, f"{args.survey} catalog")

    metadata = normalize_catalog_columns(
        pd.DataFrame(
            targets.crossmatch(
                catalog,
                radius_arcsec=args.radius_arcsec,
                n_neighbors=1,
                suffixes=("_target", f"_{suffix}"),
                suffix_method="all_columns",
            ).compute()
        ),
        args.survey,
    )
    if len(metadata) != len(targets_df):
        raise RuntimeError(
            f"Retrieved metadata for {len(metadata)} of {len(targets_df)} requested objects"
        )
    if not (
        metadata["target_object_id_target"].astype(str)
        == metadata[object_id].astype(str)
    ).all():
        raise RuntimeError("Spatial retrieval returned an object ID different from its target ID")

    default_name = objects_path.name.replace("_objects.parquet", "_metadata.parquet")
    output = Path(args.output or objects_path.with_name(default_name))
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_parquet(output, index=False)
    print(f"Wrote {len(metadata):,} metadata rows to {output}")


def command_fetch_mentions(args: argparse.Namespace) -> None:
    """Rehydrate mention fields via spatial lookup followed by exact mention_id."""
    manifest_path = Path(args.manifest)
    manifest = pd.read_parquet(manifest_path)
    required = {"mention_id_mention", "ra_mention", "dec_mention"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"{manifest_path} is missing mention keys: {missing}")

    targets_df = (
        manifest[["mention_id_mention", "ra_mention", "dec_mention"]]
        .drop_duplicates("mention_id_mention")
        .rename(
            columns={
                "mention_id_mention": "target_mention_id",
                "ra_mention": "ra",
                "dec_mention": "dec",
            }
        )
    )
    targets = lsdb.from_dataframe(targets_df, ra_column="ra", dec_column="dec")
    mentions = lsdb.open_catalog(args.mentions, columns=MENTION_COLUMNS)
    require_columns(mentions, MENTION_COLUMNS, "mentions catalog")

    # HATS is spatially indexed, not indexed by mention_id. Spatial matching
    # here only finds local candidates; identity is resolved solely by the ID.
    # lsdb.Catalog.crossmatch drops `n_neighbors=None` entirely (falls back to
    # its own default of 1), so "all candidates within radius" has to be
    # requested as a concrete, generously large neighbor count instead.
    candidates = pd.DataFrame(
        targets.crossmatch(
            mentions,
            radius_arcsec=args.radius_arcsec,
            n_neighbors=args.max_candidates,
            suffixes=("_target", "_mention"),
            suffix_method="all_columns",
        ).compute()
    )
    full_mentions = candidates.loc[
        candidates["target_mention_id_target"].astype(str)
        == candidates["mention_id_mention"].astype(str)
    ].copy()
    matched_ids = full_mentions["target_mention_id_target"].astype(str)
    missing_ids = set(targets_df["target_mention_id"].astype(str)).difference(matched_ids)
    duplicate_ids = matched_ids.duplicated()
    if missing_ids or duplicate_ids.any():
        raise RuntimeError(
            f"Could not uniquely rehydrate mentions: {len(missing_ids)} missing, "
            f"{int(duplicate_ids.sum())} duplicated"
        )

    # Carry through the input manifest's own survey-side columns (e.g.
    # object_id_legacy, ra_legacy, dec_legacy) so the output is self-
    # sufficient for joining back to an object - without this, callers had
    # to keep the input manifest/captions file around forever just to
    # recover which object each rehydrated mention belongs to. `_mention`
    # columns and `_dist_arcsec` are excluded: the former are literature
    # fields already present here (fresher, from the full rehydration
    # rather than the input's possibly-stale snapshot), the latter is this
    # step's own crossmatch distance and would collide with the input's.
    mention_key_cols = {"mention_id_mention", "ra_mention", "dec_mention"}
    passthrough_columns = [
        column
        for column in manifest.columns
        if column not in mention_key_cols and not column.endswith("_mention") and column != "_dist_arcsec"
    ]
    if passthrough_columns:
        passthrough = manifest[["mention_id_mention", *passthrough_columns]].drop_duplicates("mention_id_mention")
        full_mentions = full_mentions.merge(passthrough, on="mention_id_mention", how="left")

    default_name = f"{manifest_path.stem}_full_mentions.parquet"
    output = Path(args.output or manifest_path.with_name(default_name))
    output.parent.mkdir(parents=True, exist_ok=True)
    full_mentions.to_parquet(output, index=False)
    print(f"Wrote {len(full_mentions):,} full mention rows to {output}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    match = subparsers.add_parser("match", help="create the lightweight match manifest")
    match.add_argument("--mentions", default=MENTIONS, help="HATS URI for literature mentions")
    match.add_argument("--survey", choices=SURVEYS, default="legacy-south")
    match.add_argument("--catalog", help="override the survey's HATS URI")
    match.add_argument("--radius-arcsec", type=float, default=1.0)
    match.add_argument(
        "--output",
        help="output Parquet path (default is based on --survey)",
    )
    match.set_defaults(func=command_match)

    top_k = subparsers.add_parser("top-k", help="write the top-K caption-rich subsets")
    top_k.add_argument("manifest", help="manifest created by the match command")
    top_k.add_argument("--survey", choices=SURVEYS, default="legacy-south")
    top_k.add_argument(
        "--k",
        type=parse_selection_count,
        default=100,
        help="number of caption-rich objects, or 'all' for the full intersection",
    )
    top_k.add_argument("--output-dir", default="outputs/crossmatches")
    top_k.set_defaults(func=command_top_k)

    fetch = subparsers.add_parser("fetch-images", help="retrieve images for a selected object subset")
    fetch.add_argument("objects", help="object subset created by the top-k command")
    fetch.add_argument("--survey", choices=SURVEYS, default="legacy-south")
    fetch.add_argument("--catalog", help="override the survey's HATS URI")
    fetch.add_argument(
        "--columns",
        default=None,
        nargs="+",
        help="heavy columns to retrieve (default depends on --survey)",
    )
    fetch.add_argument("--radius-arcsec", type=float, default=0.1)
    fetch.add_argument(
        "--output",
        help="output Parquet path (default is based on --survey)",
    )
    fetch.set_defaults(func=command_fetch_images)


    metadata = subparsers.add_parser(
        "fetch-metadata",
        help="retrieve all non-image survey columns for a selected object subset",
    )
    metadata.add_argument("objects", help="object subset created by the top-k command")
    metadata.add_argument("--survey", choices=SURVEYS, default="legacy-south")
    metadata.add_argument("--catalog", help="override the survey HATS URI")
    metadata.add_argument("--radius-arcsec", type=float, default=0.1)
    metadata.add_argument(
        "--output",
        help="output Parquet path (default is derived from the object subset path)",
    )
    metadata.set_defaults(func=command_fetch_metadata)


    mention_data = subparsers.add_parser(
        "fetch-mentions",
        help="rehydrate all original mention fields for an existing manifest",
    )
    mention_data.add_argument("manifest", help="match manifest or caption subset")
    mention_data.add_argument("--mentions", default=MENTIONS, help="HATS URI for literature mentions")
    mention_data.add_argument("--radius-arcsec", type=float, default=0.1)
    mention_data.add_argument(
        "--max-candidates",
        type=int,
        default=50,
        help=(
            "upper bound on same-position mention candidates to fetch per target "
            "before resolving identity by exact mention_id (lsdb has no "
            "unlimited-neighbors mode, so this must be a concrete integer)"
        ),
    )
    mention_data.add_argument(
        "--output",
        help="output Parquet path (default is derived from the input manifest path)",
    )
    mention_data.set_defaults(func=command_fetch_mentions)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
