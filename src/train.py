"""
Train an XGBoost classifier on the combined NRC+DistilBERT feature set.

Usage:
    python src/train.py
"""

import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from evaluate import evaluate

SEED = 42
FEATURES_PATH = "data/nrc_distilbert_features.csv"
MODEL_OUTPUT_PATH = "models/xgb_distilbert_nrc.joblib"
ROC_OUTPUT_PATH = "results/roc_xgb_distilbert_nrc.json"

PARAM_DIST = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

LR_RANGE = [0.01, 0.05, 0.1, 0.2, 0.3]


def load_features(path: str = FEATURES_PATH):
    """Load the combined feature matrix and split into X, y (label-encoded)."""
    df = pd.read_csv(path)
    X = df.drop(columns=["label"])
    y_raw = df["label"]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    return X, y, le


def run_hyperparameter_search(X_train, y_train, seed: int = SEED) -> RandomizedSearchCV:
    """Randomized search over XGBoost hyperparameters, optimising macro F1."""
    search = RandomizedSearchCV(
        XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=seed, n_jobs=-1),
        PARAM_DIST, n_iter=30, cv=5,
        scoring="f1_macro", random_state=seed,
        n_jobs=-1, verbose=1, refit=True,
    )
    search.fit(X_train, y_train)
    print("Best XGBoost params:", search.best_params_)
    print(f"Best CV macro F1: {search.best_score_:.4f}")
    return search


def plot_learning_rate_curve(best_params: dict, X_train, y_train, seed: int = SEED, lr_range=LR_RANGE):
    """Sweep learning rate around the best params to visualise the F1/time trade-off."""
    cv_mean_f1, cv_std_f1, times = [], [], []

    for lr in lr_range:
        clf = XGBClassifier(
            **{**best_params, "learning_rate": lr},
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=seed, n_jobs=-1,
        )
        t0 = time.time()
        scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1)
        times.append(time.time() - t0)
        cv_mean_f1.append(scores.mean())
        cv_std_f1.append(scores.std())
        print(f"lr={lr}: {scores.mean():.4f} +/- {scores.std():.4f}")

    cv_mean_f1 = np.array(cv_mean_f1)
    cv_std_f1 = np.array(cv_std_f1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(lr_range, cv_mean_f1, marker="o", label="Mean CV F1")
    axes[0].fill_between(lr_range, cv_mean_f1 - cv_std_f1, cv_mean_f1 + cv_std_f1, alpha=0.2, label="+/-1 std")
    axes[0].set_xlabel("Learning rate")
    axes[0].set_ylabel("CV Macro F1")
    axes[0].set_title("XGBoost validation curve")
    axes[0].legend()

    axes[1].plot(lr_range, times, marker="o", color="orange")
    axes[1].set_xlabel("Learning rate")
    axes[1].set_ylabel("CV time (s)")
    axes[1].set_title("XGBoost training time")

    plt.tight_layout()
    plt.show()


def main():
    X, y, le = load_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    search = run_hyperparameter_search(X_train, y_train)
    plot_learning_rate_curve(search.best_params_, X_train, y_train)

    xgb_final = search.best_estimator_
    y_pred = xgb_final.predict(X_test)
    y_prob = xgb_final.predict_proba(X_test)

    evaluate(
        "XGBoost with DistilBERT+NRC",
        y_test, y_pred, y_prob, le.classes_,
        roc_output_path=ROC_OUTPUT_PATH,
    )

    joblib.dump({"model": xgb_final, "label_encoder": le}, MODEL_OUTPUT_PATH)
    print(f"Saved trained model to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
