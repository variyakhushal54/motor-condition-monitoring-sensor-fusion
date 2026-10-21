"""Compare Random Forest and Gradient Boosting on a validation split of the training data."""
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split

from config import CLASS_NAMES, FEATURE_COLUMNS, PROCESSED_DIR


def candidate_models():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
        ),
    }


def main():
    train = pd.read_csv(PROCESSED_DIR / "training_data.csv")
    X = train[FEATURE_COLUMNS]
    y = train["Status"].astype(int)

    X_fit, X_val, y_fit, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Fit rows: {len(X_fit)}, validation rows: {len(X_val)}")

    results = {}
    for name, model in candidate_models().items():
        model.fit(X_fit, y_fit)
        pred = model.predict(X_val)
        acc = accuracy_score(y_val, pred)
        results[name] = (acc, pred)
        print(f"{name}: validation accuracy {acc:.2%}")

    best = max(results, key=lambda name: results[name][0])
    acc, pred = results[best]
    print(f"\nBest model on validation data: {best}")
    print(f"Accuracy: {acc:.2%}")
    print(f"Weighted F1: {f1_score(y_val, pred, average='weighted'):.4f}")
    print(f"RMSE of class labels: {mean_squared_error(y_val, pred) ** 0.5:.4f}")
    print(classification_report(y_val, pred, target_names=CLASS_NAMES))


if __name__ == "__main__":
    main()
