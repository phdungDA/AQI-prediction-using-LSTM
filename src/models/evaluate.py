"""
Đánh giá model LSTM đã train.

Mặc định đánh giá trên tập VALIDATION — dùng kết quả này để so sánh
cấu hình/kiến trúc (vd LSTM vs BiLSTM vs Random Forest). Tập TEST chỉ nên
chạy MỘT LẦN DUY NHẤT, sau khi đã chốt model cuối cùng.

Chạy từ thư mục gốc project:
    python src/models/evaluate.py                # validation (mặc định)
    python src/models/evaluate.py --split test    # chỉ khi đã chốt model cuối cùng
    python src/models/evaluate.py --split both    # chạy cả val và test

Kết quả lưu vào models/:
    eval_metrics_lstm_{val|test}.json
    confusion_matrix_lstm_{val|test}.png
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from tensorflow.keras.models import load_model

from src.data.preprocess import run_preprocessing
from src.utils.eval_report import generate_evaluation_report

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "lstm_baseline.h5")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "models")
MODEL_NAME   = "lstm"


def evaluate(model_path: str = MODEL_PATH, split: str = "val"):
    """
    split="val"  (mặc định) : dùng để so sánh/chọn cấu hình, kiến trúc.
    split="test"            : chỉ chạy khi đã chốt model.
    """
    if split not in ("val", "test"):
        raise ValueError('split phải là "val" hoặc "test"')

    if split == "test":
        print("⚠️  Bạn đang đánh giá trên tập TEST.")
        print("    Chỉ làm việc này khi đã CHỐT model cuối cùng — không dùng")
        print("    kết quả test để so sánh hay chọn lại cấu hình/kiến trúc.\n")

    # ── 1. Load model + dữ liệu ───────────────────────────────────────────
    print("=" * 55)
    print(f"  [{MODEL_NAME}] Load model & dữ liệu {split.upper()}")
    print("=" * 55)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model tại '{model_path}'. Hãy chạy train.py trước.")

    model = load_model(model_path)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = run_preprocessing()

    X_eval, y_eval = (X_val, y_val) if split == "val" else (X_test, y_test)
    print(f"✅ X_{split}: {X_eval.shape} | y_{split}: {y_eval.shape}")

    # ── 2. Predict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  Dự đoán trên tập {split.upper()}")
    print("=" * 55)
    y_pred = np.argmax(model.predict(X_eval, verbose=0), axis=1)

    # ── 3. Báo cáo + lưu kết quả (hàm dùng chung) ──────────────────────────
    return generate_evaluation_report(
        y_true=y_eval, y_pred=y_pred, output_dir=OUTPUT_DIR,
        split=split, model_name=MODEL_NAME,
        title=f"Confusion Matrix - LSTM ({split})",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=["val", "test", "both"], default="val",
        help='"val" (mặc định), "test" (chỉ dùng khi đã chốt model) hoặc "both" (chạy cả hai).',
    )
    args = parser.parse_args()
    for s in (["val", "test"] if args.split == "both" else [args.split]):
        evaluate(split=s)
