"""
Đánh giá model LSTM đã train trên tập test.
Chạy từ thư mục gốc project:
    python src/models/evaluate.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

from src.data.preprocess import run_preprocessing
from src.utils.metrics import (
    compute_metrics, get_confusion_matrix, get_classification_report,
    class_distribution, AQI_LABELS,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "lstm_baseline.h5")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "models")


def evaluate(model_path: str = MODEL_PATH):
    # ── 1. Load model + dữ liệu ───────────────────────────────────────────
    print("=" * 55)
    print("  Load model & dữ liệu test")
    print("=" * 55)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model tại '{model_path}'. Hãy chạy train.py trước.")

    model = load_model(model_path)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = run_preprocessing()
    print(f"✅ X_test: {X_test.shape} | y_test: {y_test.shape}")

    # ── 2. Predict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Dự đoán trên tập test")
    print("=" * 55)
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # ── 3. Tính metrics ────────────────────────────────────────────────────
    metrics = compute_metrics(y_test, y_pred)
    print("\n📊 Chỉ số tổng quan:")
    for k, v in metrics.items():
        print(f"   {k:10}: {v:.4f}")

    print("\n📊 Phân bố lớp trong tập test (kiểm tra imbalance):")
    for k, v in class_distribution(y_test).items():
        print(f"   {k:12}: {v}")

    print("\n📊 Classification report:")
    report = get_classification_report(y_test, y_pred)
    print(report)

    # ── 4. Confusion matrix ───────────────────────────────────────────────
    cm = get_confusion_matrix(y_test, y_pred)
    print("\n📊 Confusion matrix:")
    print(cm)

    # ── 5. Lưu kết quả ─────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metrics_path = os.path.join(OUTPUT_DIR, "eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "class_distribution": class_distribution(y_test),
            "classification_report": report,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Đã lưu metrics: {metrics_path}")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=AQI_LABELS, yticklabels=AQI_LABELS, ax=ax)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title("Confusion Matrix - LSTM Baseline")
    plt.tight_layout()

    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"✅ Đã lưu confusion matrix: {cm_path}")

    return metrics, cm


if __name__ == "__main__":
    evaluate()
