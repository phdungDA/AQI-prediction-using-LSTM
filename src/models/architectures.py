"""
Baseline model: Input -> LSTM -> Dense -> Softmax -> AQI class
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_baseline(input_shape: tuple, num_classes: int = 5, lstm_units: int = 64) -> keras.Model:
    """
    input_shape: (window_size, num_features)
    num_classes: số lớp AQI (mặc định 5: 1=Good ... 5=Very Poor, đã map về 0..4)
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(lstm_units),
        layers.Dropout(0.2),  # tắt ngẫu nhiên 20% neuron mỗi bước train để giảm overfit
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def build_bilstm(input_shape: tuple, num_classes: int = 5, lstm_units: int = 64) -> keras.Model:
    """
    BiLSTM: Input -> Bidirectional(LSTM) -> Dropout -> Bidirectional(LSTM) -> Dense -> Softmax
 
    input_shape: (window_size, num_features)
    num_classes: số lớp AQI (mặc định 5, đã map về 0..4) — giống hệt
                 build_lstm_baseline, để so sánh công bằng trên cùng
                 1 bài toán, cùng 1 bộ dữ liệu (qua run_preprocessing()).
 
    Khác biệt so với build_lstm_baseline: đọc chuỗi theo CẢ 2 CHIỀU
    (xuôi + ngược) trong phạm vi window_size giờ lịch sử đã biết — giúp
    nắm bắt tốt hơn các pha "đang tăng" hay "đã qua đỉnh, đang giảm" của
    một đợt ô nhiễm, điều khó nắm bắt đầy đủ nếu chỉ đọc một chiều xuôi
    như LSTM thường. Dùng 2 lớp BiLSTM xếp chồng (lớp đầu
    return_sequences=True) để học biểu diễn phân cấp: lớp đầu bắt pattern
    cục bộ (biến động giờ-qua-giờ), lớp sau tổng hợp pattern toàn cục của
    cả cửa sổ.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True)),
        layers.Dropout(0.2),
        layers.Bidirectional(layers.LSTM(lstm_units // 2)),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
 
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
