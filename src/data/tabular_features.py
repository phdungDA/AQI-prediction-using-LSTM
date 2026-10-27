"""
Chuyển window 3D (N, window_size, n_features) từ pipeline LSTM thành
đặc trưng trễ (lag features) dạng bảng 2D (N, window_size * n_features)
để dùng cho các model classic (Logistic Regression, Random Forest, XGBoost...).

Dùng LẠI nguyên pipeline tiền xử lý ở src/data/dataloader.py (split theo
thời gian, fit scaler chỉ trên train, chống rò rỉ) — chỉ khác bước cuối:
không giữ dạng 3D cho RNN mà "duỗi thẳng" ra thành cột đặc trưng trễ.

X[:, t, f] = giá trị đặc trưng f tại (window_size - t) giờ trước thời điểm
dự đoán. Duỗi thành cột "{feature}_lag{k}" với k = window_size - t
(k=1 là giờ gần nhất, k=window_size là giờ xa nhất trong cửa sổ).
"""

import numpy as np


def flatten_windows(X: np.ndarray, feature_cols: list, window_size: int):
    """
    Tham số:
        X            : mảng (N, window_size, n_features) từ run_preprocessing()
        feature_cols : danh sách tên đặc trưng gốc, đúng thứ tự (vd FEATURE_COLS)
        window_size  : số giờ lịch sử trong mỗi window (vd WINDOW_SIZE)

    Trả về:
        X_flat       : mảng (N, window_size * n_features)
        column_names : list tên cột tương ứng, dạng "{feature}_lag{k}"
    """
    n, w, f = X.shape
    assert w == window_size, f"window_size không khớp: X có {w}, truyền vào {window_size}"
    assert f == len(feature_cols), f"số đặc trưng không khớp: X có {f}, feature_cols có {len(feature_cols)}"

    X_flat = X.reshape(n, w * f)

    column_names = []
    for t in range(w):
        lag = window_size - t  # t=0 (giờ xa nhất) -> lag lớn nhất; t=w-1 (gần nhất) -> lag1
        for feat in feature_cols:
            column_names.append(f"{feat}_lag{lag}")

    return X_flat, column_names
