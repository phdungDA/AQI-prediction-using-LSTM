"""
Các hàm tính chỉ số đánh giá cho bài toán phân loại AQI (5 lớp).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

AQI_LABELS = ["Good", "Fair", "Moderate", "Poor", "Very Poor"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> dict:
    """
    Tính accuracy, precision, recall, f1 (macro-average để không thiên vị lớp đa số).

    Tham số:
        y_true, y_pred : mảng nhãn (0..4)
        average        : kiểu trung bình cho precision/recall/f1 ("macro"/"weighted"/"micro")

    Trả về: dict các chỉ số
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
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
