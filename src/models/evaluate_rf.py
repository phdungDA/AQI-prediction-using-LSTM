"""
Đánh giá model Random Forest đã train (xem src/models/train_rf.py).

Mặc định đánh giá trên tập VALIDATION — dùng để so sánh với LSTM/BiLSTM.
Tập TEST chỉ nên chạy MỘT LẦN DUY NHẤT, sau khi đã chốt model cuối cùng.

Chạy từ thư mục gốc project:
    python src/models/evaluate_rf.py                # validation (mặc định)
    python src/models/evaluate_rf.py --split test    # chỉ khi đã chốt model cuối cùng
    python src/models/evaluate_rf.py --split both    # chạy cả val và test

Kết quả lưu vào models/randomforest/:
    eval_metrics_randomforest_{val|test}.json
    confusion_matrix_randomforest_{val|test}.png
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import joblib

from src.data.preprocess import run_preprocessing
from src.data.dataloader import FEATURE_COLS, WINDOW_SIZE
from src.data.tabular_features import flatten_windows
from src.utils.eval_report import generate_evaluation_report

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "randomforest", "rf_model.pkl")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "models", "randomforest")
MODEL_NAME   = "randomforest"


def evaluate(model_path: str = MODEL_PATH, split: str = "val"):
    if split not in ("val", "test"):
        raise ValueError('split phải là "val" hoặc "test"')

    if split == "test":
        print("⚠️  Bạn đang đánh giá trên tập TEST.")
        print("    Chỉ làm việc này khi đã CHỐT model cuối cùng — không dùng")
        print("    kết quả test để so sánh hay chọn lại cấu hình.\n")

    # ── 1. Load model + dữ liệu ───────────────────────────────────────────
    print("=" * 55)
    print(f"  [{MODEL_NAME}] Load model & dữ liệu {split.upper()}")
    print("=" * 55)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model tại '{model_path}'. Hãy chạy train_rf.py trước.")

    model = joblib.load(model_path)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = run_preprocessing()

    X_eval, y_eval = (X_val, y_val) if split == "val" else (X_test, y_test)
    X_eval_flat, _ = flatten_windows(X_eval, FEATURE_COLS, WINDOW_SIZE)
    print(f"✅ X_{split}: {X_eval_flat.shape} | y_{split}: {y_eval.shape}")

    # ── 2. Predict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  Dự đoán trên tập {split.upper()}")
    print("=" * 55)
    y_pred = model.predict(X_eval_flat)

    # ── 3. Báo cáo + lưu kết quả (hàm dùng chung) ──────────────────────────
    return generate_evaluation_report(
        y_true=y_eval, y_pred=y_pred, output_dir=OUTPUT_DIR,
        split=split, model_name=MODEL_NAME,
        title=f"Confusion Matrix - Random Forest ({split})",
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
