# Air Quality Prediction Using Time-Series Deep Learning 🌬️📈

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Deep Learning](https://img.shields.io/badge/Framework-TensorFlow-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Tổng quan dự án (Project Overview)

Dự án áp dụng các mô hình học sâu chuỗi thời gian (Time-Series Deep Learning) — LSTM, BiLSTM — để dự đoán lớp chỉ số chất lượng không khí (AQI) của giờ kế tiếp, dựa trên 24 giờ dữ liệu lịch sử các chất ô nhiễm (CO, NO, NO₂, O₃, SO₂, PM2.5, PM10, NH₃) thu thập từ OpenWeatherMap.

## 🏗️ Kiến trúc (Architecture)

```text
OpenWeatherMap ──(schedule.py, mỗi 30 phút)──▶ PostgreSQL ──▶ FastAPI (:8000) ──HTTP──▶ Streamlit
                                                                 │
                                                       LSTM model (load 1 lần lúc startup)
```

- **PostgreSQL**: lưu dữ liệu thô bảng `air_quality` (giữ tối đa ~5 năm).
- **FastAPI** (`app/api.py`): đọc DB, reindex lưới 1 giờ + nội suy gap ngắn, và phục vụ dự đoán giờ kế tiếp.
- **Streamlit** (`app/Interface/`): chỉ hiển thị, **không** truy cập DB hay load model trực tiếp mà gọi FastAPI qua HTTP.

## 📂 Cấu trúc thư mục (Directory Structure)

```text
AQI-prediction-using-LSTM/
├── data/               # Dữ liệu (Raw & Processed) - gitignore
├── models/             # Trọng số mô hình và Scalers - gitignore
├── notebooks/          # Jupyter notebooks cho EDA và thử nghiệm
├── src/
│   ├── data/           # dataloader, preprocess, dbmanager (đọc DB cho backend)
│   ├── ingestion/      # api_client, dbmanager (ghi DB), schedule (lấy dữ liệu mới)
│   ├── models/         # architectures, train, evaluate
│   └── utils/          # metrics
├── app/
│   ├── api.py          # FastAPI backend
│   └── Interface/      # Streamlit (app.py là entry point, page_*.py là các trang)
├── .env                # Cấu hình (API_KEY, DATABASE_URL, API_BASE_URL) - gitignore
├── requirements.txt    # Danh sách thư viện cần thiết
└── README.md           # Tài liệu hướng dẫn
```

## 📊 Tập dữ liệu (Dataset)

- **Nguồn**: OpenWeatherMap Air Pollution API (Hà Nội), dữ liệu theo giờ.
- **Đặc trưng (features)**: CO, NO, NO₂, O₃, SO₂, PM2.5, PM10, NH₃.
- **Target**: lớp AQI (1–5, tương ứng Good → Very Poor) của giờ kế tiếp, dự đoán từ 24 giờ lịch sử.

## 🚀 Hướng dẫn cài đặt (Installation)

### 1. Clone repository

```bash
git clone https://github.com/phdungDA/air-quality-prediction-using-deep-learning.git
cd air-quality-prediction-using-deep-learning
```

### 2. Tạo và kích hoạt môi trường ảo (virtual environment)

Dùng venv để các thư viện cài cho project này không ảnh hưởng tới Python hệ thống hoặc project khác.

```bash
python -m venv venv
```

Kích hoạt (activate) venv — tùy hệ điều hành:

```bash
# Windows (cmd)
venv\Scripts\activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

Sau khi activate, terminal sẽ hiện tiền tố `(venv)` ở đầu dòng lệnh. Khi không dùng nữa, gõ `deactivate` để thoát khỏi venv.

### 3. Cài đặt thư viện từ requirements.txt

```bash
pip install -r requirements.txt
```

`requirements.txt` gồm các nhóm chính:

| Nhóm | Thư viện | Dùng để |
| :--- | :--- | :--- |
| Data & ML | pandas, numpy, scikit-learn, tensorflow, statsmodels | Xử lý dữ liệu, huấn luyện model LSTM |
| Visualization | matplotlib, seaborn, plotly | Vẽ biểu đồ, EDA |
| Ingestion | requests, apscheduler, python-dotenv | Gọi API, lên lịch lấy dữ liệu, đọc `.env` |
| Storage | sqlalchemy, psycopg2-binary | Kết nối PostgreSQL |
| Deployment | streamlit, fastapi, uvicorn | Backend API + giao diện web |
| Experiment Tracking (tùy chọn) | mlflow | Theo dõi kết quả thử nghiệm |


### 5. Chuẩn bị dữ liệu và mô hình

- Đặt dataset vào `data/raw/air_quality_3years.csv` (hoặc chạy `python src/ingestion/api_client.py` để tải).
- Nạp dữ liệu lịch sử vào PostgreSQL (chạy 1 lần): `python src/ingestion/dbmanager.py`
- Đặt `lstm_baseline.h5` (và/hoặc `bilstm.h5`) cùng `minmax_scaler.pkl` vào thư mục `models/`, hoặc tự train bằng `python src/models/train.py`.

**6. Huấn luyện / đánh giá lại mô hình** (xem hướng dẫn chi tiết từ-đầu-đến-cuối ở mục bên dưới).

## 🧠 Chạy 2 model (LSTM & BiLSTM) từ lúc chưa chạy gì

Giả sử bạn vừa clone xong, chưa train/evaluate lần nào. Thứ tự dưới đây chạy được 2 model độc lập, không model nào ghi đè kết quả của model kia.

**Điều kiện cần trước:** đã activate venv, đã `pip install -r requirements.txt`, và có file dữ liệu thô `data/raw/air_quality_3years.csv` (tải qua `python src/ingestion/api_client.py` nếu chưa có).

**Bước 1 — Train LSTM baseline:**
python src/models/train.py
- Lần chạy **đầu tiên**: tự động tiền xử lý dữ liệu thô (`run_preprocessing()` trong `src/data/preprocess.py`) — đọc CSV, chia train/val/test (70/15/15), scale, cắt sliding window 24 giờ — rồi lưu kết quả vào `data/processed/` (các file `.npy` + `scaler.pkl`) để các lần sau khỏi làm lại.
- Train tối đa 50 epoch, có `EarlyStopping` (dừng sớm nếu 7 epoch liên tiếp val_loss không cải thiện) và `ModelCheckpoint` (chỉ giữ lại epoch có val_loss thấp nhất).
- Kết quả: `models/lstm_baseline.h5` (trọng số model tốt nhất) và `models/minmax_scaler.pkl` (scaler dùng chung cho cả 2 model).

**Bước 2 — Evaluate LSTM trên tập test:**

```bash
python src/models/evaluate.py
```

- Báo lỗi nếu chưa có `models/lstm_baseline.h5` → quay lại Bước 1.
- In ra accuracy, precision/recall/F1 từng lớp AQI, phân bố lớp trong tập test.
- Lưu `models/eval_metrics.json` và `models/confusion_matrix.png`.

**Bước 3 — Train BiLSTM:**

```bash
python src/models/train_bilstm.py
```

- Dùng lại đúng `data/processed/` đã có từ Bước 1 (không build lại từ đầu, trừ khi CSV thô mới hơn) — đảm bảo LSTM và BiLSTM train/test trên cùng 1 bộ dữ liệu để so sánh công bằng.
- Kết quả: `models/bilstm.h5` — **không** đụng vào `lstm_baseline.h5`.

**Bước 4 — Evaluate BiLSTM:**

```bash
python src/models/evaluate_bilstm.py
```

- Báo lỗi nếu chưa có `models/bilstm.h5` → quay lại Bước 3.
- Lưu `models/eval_metrics_bilstm.json` và `models/confusion_matrix_bilstm.png` (tên file riêng, không ghi đè kết quả của LSTM).

**Bước 5 (tùy chọn) — So sánh 2 model trên cùng biểu đồ:**

```bash
python src/models/train_compare.py
```

### Các endpoint của backend

| Endpoint | Mô tả |
| :--- | :--- |
| `GET /api/air-quality` | Toàn bộ dữ liệu (đã reindex 1h + nội suy gap ngắn), giờ UTC |
| `GET /api/predict/next-hour` | Dự đoán lớp AQI (1–5) của giờ kế tiếp từ 24h gần nhất |
| `GET /health` | Kiểm tra DB và model đã sẵn sàng chưa |

## 📈 Hiệu năng mô hình (Model Performance)

| Mô hình | RMSE | MAE | R2 |
| :--- | :---: | :---: | :---: |
| Baseline | Index | Index | Index |
| LSTM | Index | Index | Index |
| BiLSTM | Index | Index | Index |

## 👥 Nhóm phát triển (Contributors)

- Phan Huy Dũng
- Nguyễn Minh Tuấn
- Nguyễn Ngọc Tiến
