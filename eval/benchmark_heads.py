"""Lightweight classifiers over frozen v8 pre-adapter Q-former representations."""
from __future__ import annotations

import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from eval.benchmark_protocol import (
    DEVELOPMENT_DATASET_ID,
    DEVELOPMENT_REVISION,
    N_CV_FOLDS,
    SN_LABELS,
    YSE_DATASET_ID,
    YSE_REVISION,
    assert_dev_yse_disjoint,
    atomic_write_json,
    benchmark_metrics,
    stratified_bootstrap_intervals,
)


LABEL_TO_INDEX = {label: index for index, label in enumerate(SN_LABELS)}


class LinearProbe(nn.Module):
    """A genuine linear map from the ordered flattened query slots to three logits."""

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(64 * 384, len(SN_LABELS))

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        return self.classifier(queries.flatten(start_dim=1))


class TransformerProbe(nn.Module):
    """The accepted one-layer, pre-normalized transformer head."""

    def __init__(self) -> None:
        super().__init__()
        self.class_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.position = nn.Parameter(torch.zeros(1, 65, 384))
        layer = nn.TransformerEncoderLayer(
            d_model=384,
            nhead=6,
            dim_feedforward=768,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(384)
        self.classifier = nn.Linear(384, len(SN_LABELS))
        nn.init.normal_(self.class_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        cls = self.class_token.expand(queries.shape[0], -1, -1)
        tokens = torch.cat([cls, queries], dim=1) + self.position
        encoded = self.encoder(tokens)
        return self.classifier(self.norm(encoded[:, 0]))


def build_head(kind: str) -> nn.Module:
    if kind == "linear":
        return LinearProbe()
    if kind == "transformer":
        return TransformerProbe()
    raise ValueError("kind must be 'linear' or 'transformer'")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _targets(labels: list[str]) -> torch.Tensor:
    unknown = sorted(set(labels) - set(SN_LABELS))
    if unknown:
        raise ValueError(f"unknown labels: {unknown}")
    return torch.tensor([LABEL_TO_INDEX[label] for label in labels], dtype=torch.long)


def _class_weights(targets: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(targets[indices], minlength=len(SN_LABELS)).float()
    if (counts == 0).any():
        raise ValueError(f"a training split is missing a class: counts={counts.tolist()}")
    return len(indices) / (len(SN_LABELS) * counts)


@torch.no_grad()
def _predict(model: nn.Module, features: torch.Tensor, indices: torch.Tensor, batch_size: int, device: str) -> list[str]:
    model.eval()
    predictions: list[str] = []
    loader = DataLoader(TensorDataset(features[indices]), batch_size=batch_size, shuffle=False)
    for (batch,) in loader:
        logits = model(batch.to(device))
        predictions.extend(SN_LABELS[index] for index in logits.argmax(dim=-1).cpu().tolist())
    return predictions


def _fit_fold(
    kind: str,
    features: torch.Tensor,
    targets: torch.Tensor,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    *,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    device: str,
) -> dict:
    _seed_everything(seed)
    model = build_head(kind).to(device)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(targets, train_indices).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(features[train_indices], targets[train_indices]),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_true = [SN_LABELS[index] for index in targets[validation_indices].tolist()]

    best: dict | None = None
    epochs_without_improvement = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_features, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features.to(device)), batch_targets.to(device))
            loss.backward()
            optimizer.step()

        predictions = _predict(model, features, validation_indices, batch_size, device)
        metrics = benchmark_metrics(validation_true, predictions)
        score = (metrics["macro_f1"], metrics["balanced_accuracy"], -epoch)
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "epoch": epoch,
                "metrics": metrics,
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    assert best is not None
    return best


def _candidate_grid(kind: str) -> list[dict]:
    if kind == "linear":
        return [
            {"lr": 1e-3, "weight_decay": 0.0},
            {"lr": 3e-4, "weight_decay": 0.0},
            {"lr": 1e-3, "weight_decay": 1e-4},
            {"lr": 3e-4, "weight_decay": 1e-4},
        ]
    return [
        {"lr": 3e-4, "weight_decay": 1e-2},
        {"lr": 1e-4, "weight_decay": 1e-2},
    ]


def _cross_validate(
    kind: str,
    features: torch.Tensor,
    labels: list[str],
    folds: list[int],
    *,
    max_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    device: str,
) -> tuple[dict, list[dict]]:
    targets = _targets(labels)
    fold_tensor = torch.tensor(folds, dtype=torch.long)
    candidates = []
    for hyperparameters in _candidate_grid(kind):
        fold_results = []
        for fold in range(N_CV_FOLDS):
            validation_indices = torch.nonzero(fold_tensor == fold, as_tuple=False).flatten()
            train_indices = torch.nonzero(fold_tensor != fold, as_tuple=False).flatten()
            result = _fit_fold(
                kind,
                features,
                targets,
                train_indices,
                validation_indices,
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                seed=seed + fold,
                device=device,
                **hyperparameters,
            )
            fold_results.append({
                "fold": fold,
                "n_train": int(len(train_indices)),
                "n_validation": int(len(validation_indices)),
                "best_epoch": result["epoch"],
                "macro_f1": result["metrics"]["macro_f1"],
                "balanced_accuracy": result["metrics"]["balanced_accuracy"],
            })
        mean_macro = float(np.mean([result["macro_f1"] for result in fold_results]))
        mean_balanced = float(np.mean([result["balanced_accuracy"] for result in fold_results]))
        selected_epoch = max(1, int(round(float(np.median([r["best_epoch"] for r in fold_results])))))
        candidates.append({
            "hyperparameters": hyperparameters,
            "folds": fold_results,
            "mean_macro_f1": mean_macro,
            "mean_balanced_accuracy": mean_balanced,
            "selected_epoch": selected_epoch,
        })
    selected = max(
        candidates,
        key=lambda candidate: (
            candidate["mean_macro_f1"],
            candidate["mean_balanced_accuracy"],
            -candidate["selected_epoch"],
        ),
    )
    return selected, candidates


def _fit_all(
    kind: str,
    features: torch.Tensor,
    labels: list[str],
    *,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str,
) -> nn.Module:
    _seed_everything(seed)
    targets = _targets(labels)
    all_indices = torch.arange(len(targets))
    model = build_head(kind).to(device)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(targets, all_indices).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(features, targets), batch_size=batch_size, shuffle=True, generator=generator,
    )
    for _ in range(epochs):
        model.train()
        for batch_features, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features.to(device)), batch_targets.to(device))
            loss.backward()
            optimizer.step()
    return model


def train_and_evaluate_head(
    kind: str,
    development_path: Path,
    yse_path: Path,
    output_dir: Path,
    *,
    max_epochs: int = 100,
    patience: int = 12,
    batch_size: int = 64,
    seed: int = 20260917,
    device: str = "cuda",
    expected_protocol_fingerprint: str,
) -> dict:
    """Select by five-fold CV, retrain on all 987 development objects, then open YSE once."""
    started = time.perf_counter()
    development = torch.load(development_path, map_location="cpu", weights_only=False)
    if (
        development.get("dataset_id") != DEVELOPMENT_DATASET_ID
        or development.get("dataset_revision") != DEVELOPMENT_REVISION
    ):
        raise ValueError("development representation cache has incompatible dataset provenance")
    if development.get("protocol_fingerprint") != expected_protocol_fingerprint:
        raise ValueError("development representation cache has a stale protocol fingerprint")
    features = development["representations"].float()
    labels = list(development["labels"])
    folds = list(development["cv_folds"])
    if tuple(features.shape[1:]) != (64, 384) or len(features) != 987:
        raise ValueError(f"unexpected development representation shape {tuple(features.shape)}")
    if set(folds) != set(range(N_CV_FOLDS)):
        raise ValueError(f"development cache does not contain folds 0..{N_CV_FOLDS - 1}")

    selection_started = time.perf_counter()
    selected, candidates = _cross_validate(
        kind,
        features,
        labels,
        folds,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    selection_seconds = time.perf_counter() - selection_started
    final_started = time.perf_counter()
    final_model = _fit_all(
        kind,
        features,
        labels,
        epochs=selected["selected_epoch"],
        batch_size=batch_size,
        seed=seed + 10_000,
        device=device,
        **selected["hyperparameters"],
    )

    final_training_seconds = time.perf_counter() - final_started
    # YSE is not loaded until model selection and all-development retraining are complete.
    yse = torch.load(yse_path, map_location="cpu", weights_only=False)
    if yse.get("dataset_id") != YSE_DATASET_ID or yse.get("dataset_revision") != YSE_REVISION:
        raise ValueError("YSE representation cache has incompatible dataset provenance")
    protocol_fingerprint = development["protocol_fingerprint"]
    if yse.get("protocol_fingerprint") != expected_protocol_fingerprint:
        raise ValueError("YSE representation cache has a stale protocol fingerprint")
    yse_features = yse["representations"].float()
    yse_labels = list(yse["labels"])
    if tuple(yse_features.shape) != (266, 64, 384):
        raise ValueError(f"unexpected YSE representation shape {tuple(yse_features.shape)}")
    assert_dev_yse_disjoint(development["object_ids"], yse["object_ids"])
    yse_indices = torch.arange(len(yse_features))
    evaluation_started = time.perf_counter()
    predictions = _predict(final_model, yse_features, yse_indices, batch_size, device)
    metrics = benchmark_metrics(yse_labels, predictions)
    bootstrap_intervals = stratified_bootstrap_intervals(yse_labels, predictions)
    evaluation_seconds = time.perf_counter() - evaluation_started

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{kind}_probe.pt"
    temporary = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save({
        "kind": kind,
        "state_dict": final_model.cpu().state_dict(),
        "selected": selected,
        "seed": seed,
        "development_objects": 987,
        "protocol_fingerprint": protocol_fingerprint,
    }, temporary)
    os.replace(temporary, checkpoint_path)
    report = {
        "kind": kind,
        "protocol_fingerprint": protocol_fingerprint,
        "parameter_count": parameter_count(final_model),
        "selection": {
            "method": f"{N_CV_FOLDS}-fold stratified SHA-256 cross-validation",
            "primary": "mean_macro_f1",
            "secondary": "mean_balanced_accuracy",
            "tertiary": "earlier_selected_epoch",
            "selected": selected,
            "candidates": candidates,
        },
        "final_training": {
            "n_objects": 987,
            "epochs": selected["selected_epoch"],
            "class_weighted_cross_entropy": True,
            "seed": seed + 10_000,
            "runtime_seconds": final_training_seconds,
        },
        "yse": {
            "n_objects": len(yse_labels),
            "metrics": metrics,
            "bootstrap_95_intervals": bootstrap_intervals,
            "predictions": [
                {"object_id": oid, "true_label": true, "predicted_label": pred}
                for oid, true, pred in zip(yse["object_ids"], yse_labels, predictions)
            ],
        },
        "runtime": {
            "cross_validation_seconds": selection_seconds,
            "final_training_seconds": final_training_seconds,
            "yse_evaluation_seconds": evaluation_seconds,
            "total_seconds": time.perf_counter() - started,
        },
    }
    atomic_write_json(output_dir / f"{kind}_report.json", report)
    return report
