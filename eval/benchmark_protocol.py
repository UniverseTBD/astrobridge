"""Frozen protocol helpers for the AstroBridge v8 transient-classification benchmark.

This module is deliberately model-free.  It owns the scientific contract shared by data
preparation, all seven generative routes, the two frozen-representation heads, and scoring.
Keeping these rules in one place prevents route-specific prompt or parsing drift while leaving
the older evaluation tracks unchanged.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from eval.metrics.classification import classification_report


SN_LABELS = ("SN Ia", "SN II", "SN Ibc")
UNPARSEABLE = "UNPARSEABLE"
CONFUSION_COLUMNS = (*SN_LABELS, UNPARSEABLE)

SYSTEM_PROMPT = "You are an astronomy assistant. Describe only what the provided observation supports."
V8_USER_PROMPT = "Describe the scientifically supported properties of this observation."
BASE_USER_PROMPT = (
    "Describe the scientifically supported properties of this observation. End your response "
    "with a final sentence that classifies the transient as exactly one of: SN Ia, SN II, or SN Ibc."
)
TABLE_CONTEXT = (
    "The numerical table below is the supplied light-curve observation. TIME_DAYS is relative "
    "to the first retained row. FLUXCAL and FLUXCAL_ERR use SNANA zero point 27.5. FLUXCAL is "
    "signed difference flux. DIFF_MAG = 27.5 - 2.5 log10(abs(FLUXCAL)). DIFF_MAG_ERR is the "
    "propagated uncertainty. Flux sign remains authoritative."
)
CHART_CONTEXT = "The attached light-curve chart is the supplied observation."

MAX_NEW_TOKENS = 512
YSE_DATASET_ID = "BuildNg/astrobridge-yse-test-dataset-v2"
YSE_REVISION = "6c2e04c3dee6cbc70e4ce543bfa6dbdec35661fc"
DEVELOPMENT_DATASET_ID = "BuildNg/astrobridge-transients-dataset"
DEVELOPMENT_REVISION = "9eb915d5c025d7c98dcbe6e7d7fa98b31551a441"
V8_REPO_ID = "UniverseTBD/astrobridge-model-v8"
V8_REVISION = "ff01f90b75eda50b387fed53e2aa09b515918dcb"
ATCAT_REVISION = "399baebe003f56ff9d0c2ed80107be93cc2626f8"
PARSER_VERSION = "final-nonempty-sentence-bounded-canonical-label-v1"
PROTOCOL_SCHEMA_VERSION = 2
BOOTSTRAP_SEED = 20260917
BOOTSTRAP_RESAMPLES = 10_000
INPUT_PREPARATION_VERSION = "atcat-use-sorted-table-chart-v1"

MODEL_IDS = {
    "astrobridge_v8_prefix": V8_REPO_ID,
    "qwen35_table": "Qwen/Qwen3.5-9B",
    "qwen35_chart": "Qwen/Qwen3.5-9B",
    "gemma4_12b_table": "google/gemma-4-12B-it",
    "gemma4_12b_chart": "google/gemma-4-12B-it",
    "gemma4_31b_table": "google/gemma-4-31B-it",
    "gemma4_31b_chart": "google/gemma-4-31B-it",
}
MODEL_REVISIONS = {
    "astrobridge_v8_prefix": V8_REVISION,
    "qwen35_table": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "qwen35_chart": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "gemma4_12b_table": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
    "gemma4_12b_chart": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
    "gemma4_31b_table": "439edf5652646a0d1bd8b46bfdc1d3645761a445",
    "gemma4_31b_chart": "439edf5652646a0d1bd8b46bfdc1d3645761a445",
}
BASE_MODEL_REVISIONS = {
    "astrobridge_v8_prefix": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    **{route: revision for route, revision in MODEL_REVISIONS.items() if route != "astrobridge_v8_prefix"},
}
GENERATIVE_ROUTES = tuple(MODEL_REVISIONS)
TABLE_ROUTES = frozenset({"qwen35_table", "gemma4_12b_table", "gemma4_31b_table"})
CHART_ROUTES = frozenset({"qwen35_chart", "gemma4_12b_chart", "gemma4_31b_chart"})

# Five folds give every validation fold 22 SN Ibc examples while keeping head training cheap.
N_CV_FOLDS = 5
CV_HASH_NAMESPACE = "astrobridge-v8-head-cv-v1"

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_LABEL_PATTERNS = {
    label: re.compile(rf"(?<!\w){re.escape(label)}(?!\w)", flags=re.IGNORECASE)
    for label in SN_LABELS
}


def parse_final_sentence(answer: str) -> tuple[str, str | None]:
    """Parse one bounded canonical label from only the final non-empty sentence or line.

    The number of occurrences matters, not only the number of distinct labels.  Thus both
    ``SN Ia or SN II`` and ``SN Ia, therefore SN Ia`` are unparseable.
    """
    parts = [part.strip() for part in _SENTENCE_BOUNDARY.split(answer.strip()) if part.strip()]
    final_sentence = parts[-1] if parts else ""
    occurrences: list[str] = []
    for canonical, pattern in _LABEL_PATTERNS.items():
        occurrences.extend(canonical for _ in pattern.finditer(final_sentence))
    return final_sentence, occurrences[0] if len(occurrences) == 1 else None


def retained_observations(row) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the shared table/chart observation after the accepted-point mask and time sort."""
    mjd = np.asarray(row["lc_mjd"], dtype=float)
    flux = np.asarray(row["atcat_flux"], dtype=float)
    flux_err = np.asarray(row["atcat_flux_error"], dtype=float)
    band_id = np.asarray(row["atcat_band_id"], dtype=int)
    use = np.asarray(row["atcat_use"], dtype=bool)
    lengths = {len(mjd), len(flux), len(flux_err), len(band_id), len(use)}
    if len(lengths) != 1:
        raise ValueError(f"object={row.get('object_id', '')!r}: light-curve arrays have unequal lengths")
    if not np.any(use):
        raise ValueError(f"object={row.get('object_id', '')!r}: no atcat_use-retained observations")
    mjd, flux, flux_err, band_id = mjd[use], flux[use], flux_err[use], band_id[use]
    order = np.argsort(mjd, kind="stable")
    return mjd[order], flux[order], flux_err[order], band_id[order]


def numerical_table(row) -> str:
    """Render every retained observation as the auditable numerical baseline input."""
    mjd, flux, flux_err, band_id = retained_observations(row)
    t0 = float(mjd[0])
    bands = {1: "g", 2: "r"}
    lines = ["TIME_DAYS BAND FLUXCAL FLUXCAL_ERR DIFF_MAG DIFF_MAG_ERR"]
    for t, f, e, band in zip(mjd, flux, flux_err, band_id):
        if float(f) == 0.0:
            mag = mag_err = float("nan")
        else:
            mag = 27.5 - 2.5 * math.log10(abs(float(f)))
            mag_err = (2.5 / math.log(10.0)) * abs(float(e)) / abs(float(f))
        lines.append(
            f"{float(t) - t0:.5f} {bands.get(int(band), str(int(band)))} "
            f"{float(f):.6f} {float(e):.6f} {mag:.5f} {mag_err:.5f}"
        )
    return "\n".join(lines)


def user_prompt_for_route(route: str, table: str | None = None) -> str:
    if route == "astrobridge_v8_prefix":
        if table is not None:
            raise ValueError("AstroBridge v8 must not receive a numerical table")
        return V8_USER_PROMPT
    if route in TABLE_ROUTES:
        if table is None:
            raise ValueError(f"route={route!r} requires its prepared numerical table")
        return f"{TABLE_CONTEXT}\n\n<light_curve>\n{table}\n</light_curve>\n\n{BASE_USER_PROMPT}"
    if route in CHART_ROUTES:
        if table is not None:
            raise ValueError(f"route={route!r} is chart-only and must not receive a table")
        return f"{CHART_CONTEXT}\n\n{BASE_USER_PROMPT}"
    raise ValueError(f"unknown route {route!r}; expected one of {GENERATIVE_ROUTES}")


def full_prompt_text(user_prompt: str) -> str:
    return f"SYSTEM\n{SYSTEM_PROMPT}\n\nUSER\n{user_prompt}"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def lightcurve_preprocessing_contract(cfg) -> dict:
    """Return the exact deterministic ATCAT preprocessing settings and their stable hash."""
    modality = cfg.modalities.lightcurve
    kwargs = modality.encoder.get("kwargs", {})
    settings = {
        "sequence_length": int(modality.max_tokens),
        "detection_window_days": float(kwargs.get("detection_window_days", 30.0)),
        "detection_snr": float(kwargs.get("detection_snr", 5.0)),
        "subsample_seed": int(kwargs.get("subsample_seed", 0)),
        "atcat_revision": ATCAT_REVISION,
        "input_preparation_version": INPUT_PREPARATION_VERSION,
    }
    encoded = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**settings, "configuration_sha256": sha256_bytes(encoded)}


def benchmark_protocol_contract(cfg) -> dict:
    """Canonical scientific contract shared by every persisted benchmark artifact."""
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "prompts": {
            "system": SYSTEM_PROMPT,
            "v8_user": V8_USER_PROMPT,
            "base_user": BASE_USER_PROMPT,
            "table_context": TABLE_CONTEXT,
            "chart_context": CHART_CONTEXT,
        },
        "parser_version": PARSER_VERSION,
        "model_ids": MODEL_IDS,
        "model_revisions": MODEL_REVISIONS,
        "base_model_revisions": BASE_MODEL_REVISIONS,
        "datasets": {
            "yse": {"id": YSE_DATASET_ID, "revision": YSE_REVISION},
            "development": {"id": DEVELOPMENT_DATASET_ID, "revision": DEVELOPMENT_REVISION},
        },
        "preprocessing": lightcurve_preprocessing_contract(cfg),
        "generation": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "thinking": False,
            "stop": "model EOS or token limit",
        },
        "batching": {"gemma4_31b_batch_size": 1},
        "head_selection": {
            "folds": N_CV_FOLDS,
            "hash_namespace": CV_HASH_NAMESPACE,
            "refit_all_development_objects": True,
        },
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "stratified_by": "true_class",
            "confidence_level": 0.95,
        },
    }


def fingerprint_contract(contract: dict) -> str:
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def benchmark_protocol_fingerprint(cfg) -> str:
    return fingerprint_contract(benchmark_protocol_contract(cfg))


def validate_local_snapshot(route: str, model_path: str) -> None:
    """Reject a visibly mismatched Hugging Face snapshot before loading large weights."""
    if route not in GENERATIVE_ROUTES:
        raise ValueError(f"unknown route={route!r}")
    leaf = Path(model_path).name
    if re.fullmatch(r"[0-9a-fA-F]{40}", leaf):
        expected = BASE_MODEL_REVISIONS[route]
        if leaf.lower() != expected.lower():
            raise ValueError(
                f"route={route!r} model snapshot is {leaf}, expected pinned revision {expected}"
            )


def deterministic_cv_folds(object_ids: Iterable[str], labels: Iterable[str], n_folds: int = N_CV_FOLDS) -> dict[str, int]:
    """Assign stratified folds by SHA-256 ordering, independent of library RNG behavior."""
    pairs = [(str(oid), str(label)) for oid, label in zip(object_ids, labels)]
    if len({oid for oid, _ in pairs}) != len(pairs):
        raise ValueError("development object_id values must be unique before fold assignment")
    bad = sorted({label for _, label in pairs} - set(SN_LABELS))
    if bad:
        raise ValueError(f"unknown development labels: {bad}")
    assignments: dict[str, int] = {}
    for label in SN_LABELS:
        class_ids = [oid for oid, observed in pairs if observed == label]
        class_ids.sort(
            key=lambda oid: (
                hashlib.sha256(f"{CV_HASH_NAMESPACE}\0{oid}".encode("utf-8")).hexdigest(), oid
            )
        )
        for index, oid in enumerate(class_ids):
            assignments[oid] = index % n_folds
    return assignments


def assert_dev_yse_disjoint(development_ids: Iterable[str], yse_ids: Iterable[str]) -> None:
    overlap = sorted(set(map(str, development_ids)) & set(map(str, yse_ids)))
    if overlap:
        raise ValueError(f"development/YSE object overlap ({len(overlap)}): {overlap[:10]}")


def benchmark_metrics(y_true: list[str], y_pred: list[str | None]) -> dict:
    """Metrics with null predictions wrong and macro averages over only the three real classes."""
    base = classification_report(y_true, y_pred, list(SN_LABELS))
    matrix = {
        true: {pred: 0 for pred in CONFUSION_COLUMNS}
        for true in SN_LABELS
    }
    predicted = Counter()
    for true, pred in zip(y_true, y_pred):
        column = pred if pred in SN_LABELS else UNPARSEABLE
        matrix[true][column] += 1
        predicted[column] += 1
    recalls = [base["per_class"][label]["recall"] for label in SN_LABELS]
    return {
        **base,
        "balanced_accuracy": sum(recalls) / len(recalls),
        "parse_rate": sum(pred in SN_LABELS for pred in y_pred) / len(y_true),
        "predicted_class_counts": {column: int(predicted.get(column, 0)) for column in CONFUSION_COLUMNS},
        "confusion_matrix": {
            "true_labels": list(SN_LABELS),
            "prediction_columns": list(CONFUSION_COLUMNS),
            "values": [[matrix[true][pred] for pred in CONFUSION_COLUMNS] for true in SN_LABELS],
        },
    }


def stratified_bootstrap_intervals(
    y_true: list[str],
    y_pred: list[str | None],
    *,
    seed: int = BOOTSTRAP_SEED,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict:
    """Fixed-seed true-class-stratified percentile intervals for final YSE metrics only."""
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("bootstrap inputs must be non-empty and have equal lengths")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    unknown = sorted(set(y_true) - set(SN_LABELS))
    if unknown:
        raise ValueError(f"unknown true labels for bootstrap: {unknown}")
    class_indices = {
        label: np.asarray([index for index, value in enumerate(y_true) if value == label], dtype=int)
        for label in SN_LABELS
    }
    missing = [label for label, indices in class_indices.items() if len(indices) == 0]
    if missing:
        raise ValueError(f"bootstrap requires every true class; missing {missing}")

    rng = np.random.default_rng(seed)
    label_index = {label: index for index, label in enumerate(SN_LABELS)}
    true_encoded = np.asarray([label_index[label] for label in y_true], dtype=int)
    pred_encoded = np.asarray(
        [label_index[prediction] if prediction in label_index else len(SN_LABELS) for prediction in y_pred],
        dtype=int,
    )
    samples = {
        "accuracy": [],
        "macro_f1": [],
        "balanced_accuracy": [],
        **{f"recall::{label}": [] for label in SN_LABELS},
    }
    for _ in range(n_resamples):
        indices = np.concatenate([
            rng.choice(class_indices[label], size=len(class_indices[label]), replace=True)
            for label in SN_LABELS
        ])
        confusion = np.bincount(
            true_encoded[indices] * len(CONFUSION_COLUMNS) + pred_encoded[indices],
            minlength=len(SN_LABELS) * len(CONFUSION_COLUMNS),
        ).reshape(len(SN_LABELS), len(CONFUSION_COLUMNS))
        true_positive = np.diag(confusion[:, :len(SN_LABELS)]).astype(float)
        support = confusion.sum(axis=1).astype(float)
        predicted = confusion[:, :len(SN_LABELS)].sum(axis=0).astype(float)
        recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
        precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0)
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(true_positive),
            where=(precision + recall) > 0,
        )
        samples["accuracy"].append(float(true_positive.sum() / len(indices)))
        samples["macro_f1"].append(float(f1.mean()))
        samples["balanced_accuracy"].append(float(recall.mean()))
        for label, value in zip(SN_LABELS, recall):
            samples[f"recall::{label}"].append(float(value))

    def interval(values: list[float]) -> dict[str, float]:
        lower, upper = np.quantile(np.asarray(values, dtype=float), [0.025, 0.975])
        return {"lower": float(lower), "upper": float(upper)}

    return {
        "method": "true-class-stratified percentile bootstrap",
        "confidence_level": 0.95,
        "seed": seed,
        "resamples": n_resamples,
        "metrics": {
            "accuracy": interval(samples["accuracy"]),
            "macro_f1": interval(samples["macro_f1"]),
            "balanced_accuracy": interval(samples["balanced_accuracy"]),
            "class_recall": {
                label: interval(samples[f"recall::{label}"])
                for label in SN_LABELS
            },
        },
    }


def require_successful_generation_records(records: list[dict], route: str) -> None:
    invalid = [
        str(record.get("object_id", "<missing>"))
        for record in records
        if record.get("failure_reason") is not None or not isinstance(record.get("raw_answer"), str)
    ]
    if invalid:
        raise RuntimeError(
            f"route={route} has {len(invalid)} execution-failed or non-string latest records; "
            f"refusing to score them as model errors. First objects: {invalid[:5]}"
        )


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


@dataclass
class JsonlLedger:
    """Append-only route ledger; successful objects resume without regeneration.

    Failed attempts remain auditable but are eligible for retry.  Consumers use the last record
    for each object, preferring a later success naturally.
    """

    path: Path

    def latest(self) -> dict[str, dict]:
        records: dict[str, dict] = {}
        if not self.path.exists():
            return records
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {self.path}:{line_number}") from exc
                records[str(record["object_id"])] = record
        return records

    def completed_ids(self) -> set[str]:
        return {
            object_id for object_id, record in self.latest().items()
            if record.get("failure_reason") is None and isinstance(record.get("raw_answer"), str)
        }

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
