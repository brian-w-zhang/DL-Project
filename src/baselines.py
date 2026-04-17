#!/usr/bin/env python3
"""
Text-only TF-IDF + Logistic Regression baseline.

Usage:
  python src/baselines.py
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)

from utils import load_split, save_metrics


def run_baseline() -> None:
    train_prompts, y_train = load_split("train")
    val_prompts, y_val = load_split("val")
    test_prompts, y_test = load_split("test")

    y_train_np = np.array(y_train)
    y_val_np = np.array(y_val)
    y_test_np = np.array(y_test)

    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_prompts)
    X_val = vectorizer.transform(val_prompts)
    X_test = vectorizer.transform(test_prompts)

    print("Training logistic regression...")
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(X_train, y_train_np)

    val_scores = clf.predict_proba(X_val)[:, 1]
    test_scores = clf.predict_proba(X_test)[:, 1]
    test_preds = clf.predict(X_test)

    val_roc = roc_auc_score(y_val_np, val_scores)
    val_pr = average_precision_score(y_val_np, val_scores)
    test_roc = roc_auc_score(y_test_np, test_scores)
    test_pr = average_precision_score(y_test_np, test_scores)
    test_f1 = f1_score(y_test_np, test_preds, average="binary")

    print(f"\nBaseline (TF-IDF + LogReg)")
    print(f"  Val  ROC-AUC: {val_roc:.4f} | Val  PR-AUC: {val_pr:.4f}")
    print(f"  Test ROC-AUC: {test_roc:.4f} | Test PR-AUC: {test_pr:.4f} | F1: {test_f1:.4f}")
    print("\nClassification report (test):")
    print(classification_report(y_test_np, test_preds, target_names=["normal", "harmful"]))

    metrics = {
        "method": "tfidf_logreg",
        "val_roc_auc": float(val_roc),
        "val_pr_auc": float(val_pr),
        "test_roc_auc": float(test_roc),
        "test_pr_auc": float(test_pr),
        "test_f1_harmful": float(test_f1),
    }
    save_metrics(metrics, "baseline_tfidf")


if __name__ == "__main__":
    run_baseline()
