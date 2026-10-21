"""Train Gradient Boosting on the training split and score it on the 20 % hold-out set."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from config import CLASS_NAMES, FEATURE_COLUMNS, FIGURE_DIR, PROCESSED_DIR


def main():
    FIGURE_DIR.mkdir(exist_ok=True)
    train = pd.read_csv(PROCESSED_DIR / "training_data.csv")
    holdout = pd.read_csv(PROCESSED_DIR / "unseen_data.csv")

    model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
    )
    model.fit(train[FEATURE_COLUMNS], train["Status"].astype(int))

    y_true = holdout["Status"].astype(int)
    y_pred = model.predict(holdout[FEATURE_COLUMNS])
    acc = accuracy_score(y_true, y_pred)

    print(f"Hold-out rows: {len(holdout)}")
    print(f"Accuracy: {acc:.2%}")
    print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Hold-out confusion matrix (accuracy {acc:.2%})")
    fig.tight_layout()
    out = FIGURE_DIR / "holdout_confusion_matrix.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
