import numpy as np
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# Load the language mapping.
with open("data/languages.json", "r") as f:
    langs_dict: dict = json.load(f)

def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict:
    logits, labels = eval_pred
    predictions: np.ndarray = np.argmax(logits, axis=1)

    # Compute global metrics.
    global_accuracy = accuracy_score(labels, predictions)
    global_precision = precision_score(labels, predictions, average="weighted", zero_division=0)
    global_recall = recall_score(labels, predictions, average="weighted", zero_division=0)
    global_f1 = f1_score(labels, predictions, average="weighted", zero_division=0)

    # Use classification_report to get detailed per-class metrics.
    # The keys in the report dict for each class are the string version of the label.
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)

    # Build per-class metrics dictionary, mapping language names to their scores.
    per_class_metrics = {}
    for lang, label_id in langs_dict.items():
        key = str(label_id)
        if key in report:
            per_class_metrics[f"precision/{lang}"] = report[key]["precision"]
            per_class_metrics[f"recall/{lang}"] = report[key]["recall"]
            per_class_metrics[f"f1/{lang}"] = report[key]["f1-score"]
            per_class_metrics[f"support/{lang}"] = report[key]["support"]

    results = {
        "accuracy/avg": global_accuracy,
        "precision/avg": global_precision,
        "recall/avg": global_recall,
        "f1/avg": global_f1,
        **per_class_metrics,
    }
    return results
