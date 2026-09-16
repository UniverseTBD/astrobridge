from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
import numpy as np
from sklearn.metrics import classification_report as sklearn_classification_report
from sklearn.preprocessing import MultiLabelBinarizer


def _safe_div(num: float, den: float, fallback: float = 0.0) -> float:
    return num / den if den > 0 else fallback


def sample_precision_recall_f1(
    gt_sets: List[Set[str]],
    pred_sets: List[Set[str]],
) -> Dict[str, float]:
    precisions, recalls, f1s, jaccards = [], [], [], []

    for gt, pred in zip(gt_sets, pred_sets):
        tp = len(gt & pred)
        n_pred = len(pred)
        n_gt = len(gt)
        union = len(gt | pred)

        if n_pred == 0 and n_gt == 0:
            p, r, f1, jacc = 1.0, 1.0, 1.0, 1.0
        elif n_pred == 0:
            p, r, f1, jacc = 1.0, 0.0, 0.0, 0.0
        elif n_gt == 0:
            p, r, f1, jacc = 0.0, 1.0, 0.0, 0.0
        else:
            p = tp / n_pred
            r = tp / n_gt
            f1 = _safe_div(2 * p * r, p + r)
            jacc = _safe_div(tp, union)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        jaccards.append(jacc)

    return {
        "mean_precision": float(np.mean(precisions)) if precisions else 0.0,
        "mean_recall": float(np.mean(recalls)) if recalls else 0.0,
        "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
        "mean_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
    }


def micro_precision_recall_f1(
    gt_sets: List[Set[str]],
    pred_sets: List[Set[str]],
) -> Dict[str, float]:
    total_tp = 0
    total_pred = 0
    total_gt = 0

    for gt, pred in zip(gt_sets, pred_sets):
        total_tp += len(gt & pred)
        total_pred += len(pred)
        total_gt += len(gt)

    mic_p = _safe_div(total_tp, total_pred)
    mic_r = _safe_div(total_tp, total_gt)
    mic_f1 = _safe_div(2 * mic_p * mic_r, mic_p + mic_r)
    total_union = total_pred + total_gt - total_tp
    mic_jacc = _safe_div(total_tp, total_union)

    return {
        "micro_precision": float(mic_p),
        "micro_recall": float(mic_r),
        "micro_f1": float(mic_f1),
        "micro_jaccard": float(mic_jacc),
    }


def exact_match_rate(
    gt_sets: List[Set[str]],
    pred_sets: List[Set[str]],
) -> float:
    if not gt_sets:
        return 0.0
    return sum(1 for gt, pred in zip(gt_sets, pred_sets) if gt == pred) / len(gt_sets)


def multilabel_hamming_loss(
    gt_sets: List[Set[str]],
    pred_sets: List[Set[str]],
    labels: List[str],
) -> float:
    if not gt_sets or not labels:
        return 0.0
    mlb = MultiLabelBinarizer(classes=labels)
    mlb.fit([set(labels)])
    yt = mlb.transform(gt_sets)
    yp = mlb.transform(pred_sets)
    return float(np.mean(yt != yp))


def snr_weighted_metrics(
    gt_dicts: List[Dict[str, float]],
    pred_sets: List[Set[str]],
) -> Dict[str, float]:
    precisions, snr_recalls, snr_f1s = [], [], []

    for gt_dict, pred in zip(gt_dicts, pred_sets):
        gt = set(gt_dict.keys())
        tp = gt & pred
        n_pred = len(pred)
        n_gt = len(gt)

        if n_pred == 0 and n_gt == 0:
            p = 1.0
        elif n_pred == 0:
            p = 1.0
        elif n_gt == 0:
            p = 0.0
        else:
            p = len(tp) / n_pred

        gt_snr_total = sum(np.log1p(gt_dict.get(l, 0)) for l in gt)
        tp_snr_total = sum(np.log1p(gt_dict.get(l, 0)) for l in tp)

        if n_gt == 0 and n_pred == 0:
            rw = 1.0
        elif gt_snr_total > 0:
            rw = tp_snr_total / gt_snr_total
        else:
            rw = 0.0

        f1w = _safe_div(2 * p * rw, p + rw)
        precisions.append(p)
        snr_recalls.append(rw)
        snr_f1s.append(f1w)

    return {
        "mean_precision": float(np.mean(precisions)) if precisions else 0.0,
        "mean_snr_weighted_recall": float(np.mean(snr_recalls)) if snr_recalls else 0.0,
        "mean_snr_weighted_f1": float(np.mean(snr_f1s)) if snr_f1s else 0.0,
    }


def micro_snr_weighted_metrics(
    gt_dicts: List[Dict[str, float]],
    pred_sets: List[Set[str]],
) -> Dict[str, float]:
    total_tp = 0
    total_pred = 0
    total_gt_snr = 0.0
    total_tp_snr = 0.0

    for gt_dict, pred in zip(gt_dicts, pred_sets):
        gt = set(gt_dict.keys())
        tp = gt & pred
        total_tp += len(tp)
        total_pred += len(pred)
        total_gt_snr += sum(np.log1p(gt_dict.get(l, 0)) for l in gt)
        total_tp_snr += sum(np.log1p(gt_dict.get(l, 0)) for l in tp)

    mic_p = _safe_div(total_tp, total_pred)
    mic_rw = _safe_div(total_tp_snr, total_gt_snr)
    mic_f1w = _safe_div(2 * mic_p * mic_rw, mic_p + mic_rw)

    return {
        "micro_snr_weighted_recall": float(mic_rw),
        "micro_snr_weighted_f1": float(mic_f1w),
    }


def per_label_report(
    gt_sets: List[Set[str]],
    pred_sets: List[Set[str]],
    labels: List[str],
    gt_dicts: Optional[List[Dict[str, float]]] = None,
) -> Dict[str, Dict[str, Any]]:
    mlb = MultiLabelBinarizer(classes=labels)
    mlb.fit([set(labels)])
    yt = mlb.transform(gt_sets)
    yp = mlb.transform(pred_sets)

    result = {}
    for i, label in enumerate(labels):
        t_mask = yt[:, i] == 1
        p_mask = yp[:, i] == 1

        tp = int((t_mask & p_mask).sum())
        fp = int((~t_mask & p_mask).sum())
        fn = int((t_mask & ~p_mask).sum())
        support = int(t_mask.sum())

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0

        entry = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

        if gt_dicts is not None:
            snrs = [d.get(label, 0.0) for j, d in enumerate(gt_dicts) if t_mask[j]]
            det_snrs = [d.get(label, 0.0) for j, d in enumerate(gt_dicts) if t_mask[j] and p_mask[j]]
            entry["mean_snr"] = float(np.mean(snrs)) if snrs else 0.0
            entry["mean_detected_snr"] = float(np.mean(det_snrs)) if det_snrs else 0.0

        result[label] = entry

    return result


def macro_label_f1(
    gt_sets: List[Set[str]],
    pred_sets: List[Set[str]],
    labels: List[str],
) -> float:
    report = per_label_report(gt_sets, pred_sets, labels)
    f1s = [report[label]["f1"] for label in labels]
    return float(np.mean(f1s)) if f1s else 0.0


def multilabel_report(
    gt_dicts: List[Dict[str, float]],
    pred_sets: List[Set[str]],
    labels: List[str],
    *,
    n_format_errors: int = 0,
) -> Dict[str, Any]:
    gt_sets = [set(d.keys()) for d in gt_dicts]

    total_samples = len(gt_dicts) + n_format_errors
    exact = exact_match_rate(gt_sets, pred_sets)
    sample = sample_precision_recall_f1(gt_sets, pred_sets)
    micro = micro_precision_recall_f1(gt_sets, pred_sets)
    snr = snr_weighted_metrics(gt_dicts, pred_sets)
    micro_snr = micro_snr_weighted_metrics(gt_dicts, pred_sets)
    plr = per_label_report(gt_sets, pred_sets, labels, gt_dicts=gt_dicts)
    macro_f1 = macro_label_f1(gt_sets, pred_sets, labels)
    hl = multilabel_hamming_loss(gt_sets, pred_sets, labels)

    return {
        "total_samples": total_samples,
        "evaluated_samples": len(gt_dicts),
        "exact_matches": int(exact * len(gt_sets)) if gt_sets else 0,
        "exact_match_rate": exact,
        "hamming_loss": hl,
        "format_errors": n_format_errors,
        "sample_level": sample,
        "dataset_micro_level": {
            **micro,
            **micro_snr,
        },
        "snr_weighted": {
            "mean_snr_weighted_recall": snr["mean_snr_weighted_recall"],
            "mean_snr_weighted_f1": snr["mean_snr_weighted_f1"],
        },
        "dataset_macro_level": {
            "mean_line_f1": macro_f1,
        },
        "per_line_metrics": plr,
    }
