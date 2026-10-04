"""Evaluation function copied verbatim from Ines shared notebook (cell 17)."""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

def evaluate_predictions(y_true, y_pred, model_name, setting, show_matrix=True):
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    row = {
        "model": model_name,
        "setting": setting,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f_macro,
        "precision_0": precision[0],
        "recall_0": recall[0],
        "f1_0": f1[0],
        "support_0": support[0],
        "precision_1": precision[1],
        "recall_1": recall[1],
        "f1_1": f1[1],
        "support_1": support[1],
    }

    summary = pd.DataFrame([row])

    if show_matrix:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Non-offensive (0)", "Offensive (1)"],
        )
        disp.plot(values_format="d")
        plt.title(f"{model_name} — {setting}")
        plt.tight_layout()
        plt.show()

    return summary
