import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)


def get_test_prediction(
    model_dir: str,
    test_csv_path: str,
    text_col: str = "review",
    label_col: str = "label",   # can be "sentiment" (pos/neg strings) or "label" (0/1)
    max_length: int = 256,
    batch_size: int = 64
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    df = pd.read_csv(test_csv_path)
    y_true = df[label_col].astype(int).to_numpy()
    texts = df[text_col].astype(str).tolist()

    # --- Batched inference ---
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            logits = model(**enc).logits  # (B, 2)
            all_logits.append(logits.detach().cpu())

    logits = torch.cat(all_logits, dim=0).numpy()

    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    y_score = probs[:, 1]                    # prob positive

    return y_true, y_score


def plot_eval_dashboard(
    y_true,
    y_score,
    threshold: float = 0.5,
    title: str = "Evaluation Dashboard",
    out_path: str = "eval_dashboard.png",
):
    """
    y_true: array-like shape (N,), values {0,1}
    y_score: array-like shape (N,), probability for positive class in [0,1]
    threshold: decision threshold for y_pred
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    y_pred = (y_score >= threshold).astype(int)

    #accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Confusion matrices
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    # ROC
    fpr, tpr, roc_th = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # PR
    prec, rec, pr_th = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    # Threshold sweep for F1/P/R
    ths = np.linspace(0.01, 0.99, 99)
    f1s, ps, rs = [], [], []
    for th in ths:
        yp = (y_score >= th).astype(int)
        f1s.append(f1_score(y_true, yp))
        ps.append(precision_score(y_true, yp, zero_division=0))
        rs.append(recall_score(y_true, yp, zero_division=0))
    f1s = np.array(f1s); ps = np.array(ps); rs = np.array(rs)
    best_idx = int(np.argmax(f1s))
    best_th = float(ths[best_idx])

    # Score distributions
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]

    # ---------------- Plot layout ----------------
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=16)

    # 1) Confusion matrix (counts)
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax1,
        xticklabels=["NEG", "POS"], yticklabels=["NEG", "POS"]
    )
    ax1.set_title(f"Confusion Matrix (counts)\nthreshold={threshold:.2f}")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")

    # 2) Confusion matrix (normalized)
    ax2 = plt.subplot2grid((2, 3), (0, 1))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=ax2,
        xticklabels=["NEG", "POS"], yticklabels=["NEG", "POS"]
    )
    ax2.set_title("Confusion Matrix (row-normalized)")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")

    # 3) ROC curve
    ax3 = plt.subplot2grid((2, 3), (0, 2))
    ax3.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    ax3.plot([0, 1], [0, 1], linestyle="--")
    ax3.set_title("ROC Curve")
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.legend(loc="lower right")

    # 4) Precision-Recall curve
    ax4 = plt.subplot2grid((2, 3), (1, 0))
    ax4.plot(rec, prec, label=f"AP={ap:.4f}")
    ax4.set_title("Precision–Recall Curve")
    ax4.set_xlabel("Recall")
    ax4.set_ylabel("Precision")
    ax4.legend(loc="lower left")

    # 5) Score distributions
    ax5 = plt.subplot2grid((2, 3), (1, 1))
    # Using seaborn histplot for nice bins; KDE optional (can be slow)
    sns.histplot(neg_scores, bins=30, stat="density", ax=ax5, label="True NEG", alpha=0.5)
    sns.histplot(pos_scores, bins=30, stat="density", ax=ax5, label="True POS", alpha=0.5)
    ax5.axvline(threshold, linestyle="--", label=f"th={threshold:.2f}")
    ax5.set_title("Predicted P(POS) distribution")
    ax5.set_xlabel("P(POS)")
    ax5.set_ylabel("Density")
    ax5.legend()

    # 6) Threshold vs F1/Precision/Recall
    ax6 = plt.subplot2grid((2, 3), (1, 2))
    ax6.plot(ths, f1s, label="F1")
    ax6.plot(ths, ps, label="Precision")
    ax6.plot(ths, rs, label="Recall")
    ax6.axvline(best_th, linestyle="--", label=f"best F1 th={best_th:.2f}")
    ax6.axvline(threshold, linestyle=":", label=f"chosen th={threshold:.2f}")
    ax6.set_title("Metrics vs threshold")
    ax6.set_xlabel("Threshold")
    ax6.set_ylabel("Score")
    ax6.set_ylim(0, 1.01)
    ax6.legend(loc="best")

    plt.tight_layout()
    plt.show()
    plt.savefig(out_path, dpi=160)
    print("Accuracy:", accuracy)

    return {
        "Accuracy": float(accuracy),
        "auc": float(roc_auc),
        "ap": float(ap),
        "best_f1": float(f1s[best_idx]),
        "best_threshold": best_th,
    }
