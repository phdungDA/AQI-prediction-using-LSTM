"""
Train baseline Random Forest với đặc trưng trễ (lag features), dùng để so sánh
với LSTM/BiLSTM trên CÙNG một bộ dữ liệu, cùng cách chia tập (xem
Huong_dan_cai_tien_AQI_LSTM — Bước 3: "Nếu phân loại ... Random Forest hoặc
XGBoost với đặc trưng trễ — Kiểm tra LSTM có vượt mô hình đơn giản hơn không").

Dùng LẠI nguyên run_preprocessing() (split theo thời gian, fit scaler chỉ trên
train, chống rò rỉ) — chỉ "duỗi" window 3D thành bảng lag features 2D qua
src/data/tabular_features.flatten_windows().

GHI CHÚ VỀ OVERFIT (đã quan sát): quét theo n_estimators (số cây) cho thấy
train accuracy ~0.99 nhưng val accuracy ~0.92 NGAY TỪ n_estimators nhỏ, và
khoảng cách đó không thu hẹp khi tăng số cây. Nguyên nhân KHÔNG nằm ở số cây,
mà ở việc mỗi cây mặc định không giới hạn độ sâu (max_depth=None) nên TỪNG
CÂY đã tự overfit ngay từ đầu — tăng n_estimators chỉ lấy trung bình nhiều
cây overfit độc lập (giảm variance) chứ không giảm được overfit gốc.
→ Sửa đúng chỗ: quét max_depth (độ sâu tối đa mỗi cây) để tìm điểm "vừa đủ
phức tạp" trước khi train accuracy tách xa val accuracy, thay vì quét
n_estimators.

Chạy từ thư mục gốc project:
    python src/models/train_rf.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.data.preprocess import run_preprocessing
from src.data.dataloader import FEATURE_COLS, WINDOW_SIZE
from src.data.tabular_features import flatten_windows

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models", "randomforest")

N_ESTIMATORS_FINAL = 300    # số cây cho model cuối cùng — KHÔNG phải nguồn gây overfit, giữ cố định
MIN_SAMPLES_LEAF   = 5      # mỗi lá tối thiểu 5 mẫu — tránh cây chia quá vụn
MIN_SAMPLES_SPLIT  = 10
RANDOM_STATE        = 42

# Dải max_depth để quét — None (không giới hạn) đưa vào cuối cùng để thấy rõ
# đối chứng "overfit nặng nhất" so với các mức có giới hạn.
MAX_DEPTH_LIST = (3, 5, 8, 10, 12, 15, 20, 30, None)


def plot_feature_importance(model: RandomForestClassifier, column_names: list, out_dir: str, top_n: int = 20):
    """
    Biểu đồ 1: Top N đặc trưng trễ quan trọng nhất theo Random Forest.
    Giúp xem model đang dựa nhiều vào giờ gần nhất (lag1-2) hay cả cửa sổ xa
    hơn (lag5-6) — và đặc trưng nào (pm2_5, pm10...) đóng góp nhiều nhất.
    """
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(8, max(5, top_n * 0.3)))
    plt.barh(
        [column_names[i] for i in order][::-1],
        importances[order][::-1],
    )
    plt.xlabel("Feature importance")
    plt.title(f"Random Forest — Top {top_n} đặc trưng trễ quan trọng nhất")
    plt.tight_layout()

    path = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ Đã lưu biểu đồ feature importance: {path}")


def plot_train_val_by_depth(X_train, y_train, X_val, y_val, out_dir: str,
                             max_depth_list=MAX_DEPTH_LIST,
                             n_estimators: int = N_ESTIMATORS_FINAL):
    """
    Biểu đồ 2: Accuracy train vs val theo max_depth (độ sâu tối đa mỗi cây) —
    thay cho quét theo n_estimators, vì nguồn gây overfit của RF nằm ở độ
    sâu/độ phức tạp từng cây, không phải số lượng cây (xem ghi chú đầu file).

    n_estimators giữ CỐ ĐỊNH ở mức đủ lớn (300) trong suốt lần quét này —
    chỉ thay đổi max_depth — để tách biệt đúng 1 biến đang kiểm tra.

    Trả về: max_depth cho val accuracy cao nhất (dùng để train model cuối).
    """
    train_scores, val_scores = [], []
    for depth in max_depth_list:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=depth,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            min_samples_split=MIN_SAMPLES_SPLIT,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)
        train_scores.append(accuracy_score(y_train, model.predict(X_train)))
        val_scores.append(accuracy_score(y_val, model.predict(X_val)))
        depth_label = depth if depth is not None else "None"
        print(f"   max_depth={str(depth_label):>4} | train acc={train_scores[-1]:.4f} | val acc={val_scores[-1]:.4f}")

    # Trục x: None không vẽ được trực tiếp -> gán vị trí sau giá trị lớn nhất,
    # chỉ để hiển thị, không ảnh hưởng tới việc chọn max_depth tốt nhất.
    finite_depths = [d for d in max_depth_list if d is not None]
    plot_x = [d if d is not None else max(finite_depths) + 5 for d in max_depth_list]
    tick_labels = [str(d) if d is not None else "None" for d in max_depth_list]

    plt.figure(figsize=(7, 5))
    plt.plot(plot_x, train_scores, marker="o", label="train accuracy")
    plt.plot(plot_x, val_scores, marker="o", label="val accuracy")
    plt.xticks(plot_x, tick_labels)
    plt.xlabel("max_depth (độ sâu tối đa mỗi cây)")
    plt.ylabel("Accuracy")
    plt.title(f"Random Forest — Accuracy train vs validation theo max_depth\n(n_estimators={n_estimators} cố định)")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, "train_val_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ Đã lưu biểu đồ train vs val theo max_depth: {path}")

    best_idx = int(np.argmax(val_scores))
    best_depth = max_depth_list[best_idx]
    print(f"\n   → max_depth cho val accuracy cao nhất: {best_depth} (val acc={val_scores[best_idx]:.4f})")
    return best_depth


def main():
    # ── 1. Load dữ liệu (dùng lại pipeline LSTM) ─────────────────────────────
    print("=" * 55)
    print("  BƯỚC 1 — Chuẩn bị dữ liệu")
    print("=" * 55)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = run_preprocessing()

    X_train_flat, column_names = flatten_windows(X_train, FEATURE_COLS, WINDOW_SIZE)
    X_val_flat, _              = flatten_windows(X_val,   FEATURE_COLS, WINDOW_SIZE)
    print(f"✅ Đặc trưng trễ: {X_train_flat.shape[1]} cột "
          f"({WINDOW_SIZE} giờ x {len(FEATURE_COLS)} đặc trưng)")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── 2. Quét max_depth để tìm điểm vừa đủ phức tạp (chống overfit) ───────
    print("\n" + "=" * 55)
    print("  BƯỚC 2 — Quét max_depth: train vs val")
    print("=" * 55)
    best_depth = plot_train_val_by_depth(X_train_flat, y_train, X_val_flat, y_val, MODEL_DIR)

    # ── 3. Train model cuối cùng với max_depth tốt nhất ──────────────────
    print("\n" + "=" * 55)
    print(f"  BƯỚC 3 — Train model cuối cùng "
          f"(n_estimators={N_ESTIMATORS_FINAL}, max_depth={best_depth})")
    print("=" * 55)
    final_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS_FINAL,
        max_depth=best_depth,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        min_samples_split=MIN_SAMPLES_SPLIT,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    final_model.fit(X_train_flat, y_train)

    train_acc = accuracy_score(y_train, final_model.predict(X_train_flat))
    val_acc   = accuracy_score(y_val,   final_model.predict(X_val_flat))
    print(f"   Train accuracy (model cuối): {train_acc:.4f}")
    print(f"   Val accuracy   (model cuối): {val_acc:.4f}")

    # ── 4. Biểu đồ feature importance ────────────────────────────────────
    print("\n" + "=" * 55)
    print("  BƯỚC 4 — Feature importance")
    print("=" * 55)
    plot_feature_importance(final_model, column_names, MODEL_DIR)

    # ── 5. Lưu model ─────────────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, "rf_model.pkl")
    joblib.dump(final_model, model_path)
    print(f"\n✅ Model đã lưu: {model_path}")
    print(f"   (scaler dùng chung với LSTM: models/minmax_scaler.pkl, không cần lưu lại)")

    return final_model


if __name__ == "__main__":
    main()
