"""
Script train BiLSTM.
Chạy từ thư mục gốc project:
    python src/models/train_bilstm.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import joblib
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from src.data.preprocess import run_preprocessing
from src.models.architectures import build_bilstm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")


def main():
    # ── 1. Load / build data ──────────────────────────────────────────────────
    print("=" * 55)
    print("  BƯỚC 1 — Chuẩn bị dữ liệu")
    print("=" * 55)
    # Dùng lại ĐÚNG run_preprocessing() mà train.py (LSTM) đang dùng —
    # đảm bảo BiLSTM và LSTM được train/test trên CÙNG 1 bộ dữ liệu,
    # cùng 1 cách chia train/val/test, cùng 1 scaler → so sánh công bằng
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = run_preprocessing()

    # ── 2. Xây model ──────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  BƯỚC 2 — Khởi tạo model")
    print("=" * 55)
    input_shape = (X_train.shape[1], X_train.shape[2])   # (24, 8)
    model = build_bilstm(input_shape=input_shape)
    model.summary()

    # ── 3. Train ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  BƯỚC 3 — Train")
    print("=" * 55)
    os.makedirs(MODEL_DIR, exist_ok=True)
    # Lưu riêng "bilstm.h5" — KHÔNG đụng vào "lstm_baseline.h5" của LSTM
    best_model_path = os.path.join(MODEL_DIR, "bilstm.h5")

    callbacks = [
        ModelCheckpoint(
            best_model_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
    )

    # ── 4. Đánh giá sơ bộ trên tập test ──────────────────────────────────────
    print("\n" + "=" * 55)
    print("  BƯỚC 4 — Đánh giá trên tập TEST")
    print("=" * 55)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test loss     : {test_loss:.4f}")
    print(f"   Test accuracy : {test_acc:.4f}")

    # ── 5. Lưu scaler ─────────────────────────────────────────────────────────
    # Dùng chung "minmax_scaler.pkl" với LSTM — scaler này được fit từ
    # run_preprocessing() ở Bước 1, không phụ thuộc kiến trúc model nào
    # tiêu thụ nó sau đó, nên LSTM và BiLSTM hoàn toàn dùng chung được.
    # Ghi lại (dù nội dung thường giống hệt) để file này tự đủ, không
    # bắt buộc phải chạy train.py trước mới có scaler.
    scaler_path = os.path.join(MODEL_DIR, "minmax_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"\n✅ Model đã lưu  : {best_model_path}")
    print(f"✅ Scaler đã lưu : {scaler_path}")

    return history


if __name__ == "__main__":
    main()