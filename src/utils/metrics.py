"""
Các hàm tính chỉ số đánh giá cho bài toán phân loại AQI (5 lớp).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

AQI_LABELS = ["Good", "Fair", "Moderate", "Poor", "Very Poor"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Tính đủ bộ chỉ số cho bài toán phân loại AQI 5 lớp, có mất cân bằng lớp:
      - accuracy          : tỉ lệ đúng thô (dễ bị lớp đa số chi phối, chỉ tham khảo)
      - balanced_accuracy  : trung bình recall từng lớp — phản ánh đúng việc model
                             có đoán tốt cả lớp hiếm (vd Very Poor) hay chỉ ăn điểm
                             nhờ đoán đúng lớp đa số
      - precision/recall_macro : trung bình không trọng số qua các lớp (mỗi lớp quan
                             trọng như nhau, bất kể số lượng mẫu)
      - f1_macro           : macro-F1 — chỉ số chính để so sánh model khi lớp lệch
      - f1_weighted        : weighted-F1 — trung bình F1 có trọng số theo support,
                             phản ánh hiệu năng tổng thể trên phân bố lớp thực tế

    Tham số:
        y_true, y_pred : mảng nhãn (0..4)

    Trả về: dict các chỉ số
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list = None) -> np.ndarray:
    """Ma trận nhầm lẫn: hàng = nhãn thật, cột = nhãn dự đoán."""
    if labels is None:
        labels = list(range(len(AQI_LABELS)))
    return confusion_matrix(y_true, y_pred, labels=labels)


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray, target_names: list = None) -> str:
    """Báo cáo chi tiết precision/recall/f1/support cho từng lớp AQI."""
    if target_names is None:
        target_names = AQI_LABELS
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


def class_distribution(y: np.ndarray) -> dict:
    """Đếm số mẫu mỗi lớp — dùng để kiểm tra class imbalance."""
    unique, counts = np.unique(y, return_counts=True)
    return {AQI_LABELS[int(u)]: int(c) for u, c in zip(unique, counts)}
