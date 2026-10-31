import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, average="binary", zero_division=0)),
    }

def forgetting(before_accuracy: float, after_accuracy: float) -> float:
    """Positive means English accuracy dropped after Hindi training."""
    return float(before_accuracy - after_accuracy)
