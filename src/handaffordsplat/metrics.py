"""Metrics matching the thesis evaluation implementation."""

from __future__ import annotations

from typing import Dict

import numpy as np


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores)
    if y_true.sum() == 0 or (~y_true).sum() == 0:
        return float("nan")
    order = np.argsort(-scores)
    ranked = y_true[order]
    true_positives = np.cumsum(ranked)
    precision = true_positives / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / y_true.sum())


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores)
    positives = int(y_true.sum())
    negatives = int((~y_true).sum())
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end

    rank_sum = ranks[y_true].sum()
    return float(
        (rank_sum - positives * (positives + 1) / 2.0)
        / max(positives * negatives, 1)
    )


def binary_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(y_true) != len(scores):
        raise ValueError("Label and score lengths differ")

    prediction = scores >= float(threshold)
    tp = int(np.logical_and(prediction, y_true).sum())
    tn = int(np.logical_and(~prediction, ~y_true).sum())
    fp = int(np.logical_and(prediction, ~y_true).sum())
    fn = int(np.logical_and(~prediction, y_true).sum())
    n = max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    return {
        "num_gaussians": float(len(y_true)),
        "iou": float(tp / max(tp + fp + fn, 1)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((tp + tn) / n),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "roc_auc": roc_auc(y_true, scores),
        "auprc": average_precision(y_true, scores),
        "gt_handle_ratio": float(y_true.mean()) if len(y_true) else 0.0,
        "pred_handle_ratio": float(prediction.mean()) if len(y_true) else 0.0,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }
