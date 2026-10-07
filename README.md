# Air Quality Prediction Using Time-Series Deep Learning 🌬️📈

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Deep Learning](https://img.shields.io/badge/Framework-PyTorch%20%2F%20TensorFlow-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Tổng quan dự án (Project Overview)
Dự án áp dụng các mô hình học sâu chuỗi thời gian (Time-Series Deep Learning) như LSTM, GRU, BiLSTM để dự đoán chỉ số chất lượng không khí (AQI) và nồng độ bụi mịn PM2.5 dựa trên dữ liệu lịch sử và các yếu tố khí tượng học. 

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
├── data/               # Dữ liệu (Raw & Processed) - Đã được gitignore
├── models/             # Trọng số mô hình và Scalers - Đã được gitignore
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
Nguồn: OpenWeatherMap Air Pollution API (Hà Nội), dữ liệu theo giờ.

Đặc trưng (Features): CO, NO, NO₂, O₃, SO₂, PM2.5, PM10, NH₃.

Target: lớp AQI (1–5) của giờ kế tiếp, dự đoán từ 24 giờ lịch sử.

## 🚀 Hướng dẫn cài đặt (Installation)
1. Clone repository:
```text
git clone https://github.com/phdungDA/air-quality-prediction-using-deep-learning.git
cd air-quality-prediction-using-deep-learning
```

2. Tạo môi trường ảo và cài đặt thư viện:
```text
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Tạo file `.env` ở thư mục gốc (không commit lên git):
```text
API_KEY = <openweathermap_api_key>
latitude = 21.0285
longtitude = 105.8542
API_BASE_URL = http://localhost:8000
DATABASE_URL = postgresql://<user>:<password>@localhost:5432/<dbname>
```

4. Chuẩn bị dữ liệu và mô hình:
- Đặt dataset vào `data/raw/air_quality_3years.csv` (hoặc chạy `python src/ingestion/api_client.py` để tải).
- Nạp dữ liệu lịch sử vào PostgreSQL (chạy 1 lần): `python src/ingestion/dbmanager.py`
- Đặt `lstm_baseline.h5` và `minmax_scaler.pkl` vào thư mục `models/` (hoặc tự train bằng `python src/models/train.py`).

## ⚙️ Hướng dẫn sử dụng (Usage)
Chạy tất cả lệnh từ **thư mục gốc project**. Web app cần **2 process** chạy song song (mỗi process một terminal), PostgreSQL phải đang chạy:

1. Khởi chạy FastAPI backend (model được load 1 lần lúc khởi động):
```text
uvicorn app.api:app --port 8000
```
Kiểm tra nhanh: `http://localhost:8000/health` · Swagger UI: `http://localhost:8000/docs`

2. Khởi chạy giao diện Streamlit:
```text
streamlit run app/Interface/app.py
```
Streamlit đọc địa chỉ backend từ `API_BASE_URL` trong `.env` (mặc định `http://localhost:8000`).

3. (Tùy chọn) Tự động lấy dữ liệu mới mỗi 30 phút vào PostgreSQL:
```text
python src/ingestion/schedule.py          # chạy liên tục
python src/ingestion/schedule.py --once   # chạy 1 lần để test
```

4. Huấn luyện / đánh giá lại mô hình:
```text
python src/models/train.py
python src/models/evaluate.py
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
| LSTM |  Index | Index | Index |
| GRU | Index | Index | Index |

## 👥 Nhóm phát triển (Contributors)

- Phan Huy Dũng
- Nguyễn Minh Tuấn 
- Nguyễn Ngọc Tiến