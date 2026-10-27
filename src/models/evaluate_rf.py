"""
Đánh giá model Random Forest đã train (xem src/models/train_rf.py).

Mặc định đánh giá trên tập VALIDATION — dùng để so sánh với LSTM/BiLSTM.
Tập TEST chỉ nên chạy MỘT LẦN DUY NHẤT, sau khi đã chốt model cuối cùng.

Dùng lại src/utils/eval_report.generate_evaluation_report() — cùng hàm có
thể tái sử dụng cho mọi model khác (XGBoost, Logistic Regression...) miễn
là có y_true/y_pred dạng nhãn, không cần viết lại logic in/lưu/vẽ confusion
matrix riêng cho từng file evaluate.

Chạy từ thư mục gốc project:
    python src/models/evaluate_rf.py                # đánh giá trên validation (mặc định)
    python src/models/evaluate_rf.py --split test    # chỉ chạy khi đã chốt model cuối cùng
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


def evaluate(model_path: str = MODEL_PATH, split: str = "val"):
    if split not in ("val", "test"):
        raise ValueError('split phải là "val" hoặc "test"')

    if split == "test":
        print("⚠️  Bạn đang đánh giá trên tập TEST.")
        print("    Chỉ làm việc này khi đã CHỐT model cuối cùng — không dùng")
        print("    kết quả test để so sánh hay chọn lại cấu hình.\n")

    # ── 1. Load model + dữ liệu ───────────────────────────────────────────
    print("=" * 55)
    print(f"  Load model & dữ liệu {split.upper()}")
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

    # ── 3. Báo cáo (dùng hàm chung) ─────────────────────────────────────────
    metrics, cm = generate_evaluation_report(
        y_true=y_eval,
        y_pred=y_pred,
        output_dir=OUTPUT_DIR,
        prefix=split,
        title=f"Confusion Matrix - Random Forest ({split})",
    )

    return metrics, cm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=["val", "test"], default="val",
        help='Tập dữ liệu để đánh giá: "val" (mặc định) hoặc "test" (chỉ dùng 1 lần).',
    )
    args = parser.parse_args()
    evaluate(split=args.split)
