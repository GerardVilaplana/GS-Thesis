import numpy as np

from handaffordsplat.metrics import binary_metrics


def test_binary_metrics_perfect_prediction():
    labels = np.array([0, 0, 1, 1], dtype=np.uint8)
    scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    metrics = binary_metrics(labels, scores, 0.5)
    for key in ["iou", "precision", "recall", "f1", "accuracy", "balanced_accuracy"]:
        assert metrics[key] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["auprc"] == 1.0
