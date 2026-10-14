"""
FastAPI backend cho dự án dự đoán AQI.

Chạy từ thư mục gốc project:
    uvicorn app.api:app --reload --port 8000
Swagger UI (thử endpoint trên trình duyệt): http://localhost:8000/docs

Endpoint:
    GET /api/air-quality        toàn bộ dữ liệu (đã reindex 1h + nội suy gap ngắn), giờ UTC
    GET /api/predict/next-hour  dự đoán lớp AQI (1..5) của giờ kế tiếp từ 24h gần nhất
    GET /health                 kiểm tra DB + model đã sẵn sàng

Model + scaler chỉ load 1 lần lúc khởi động (lifespan), không load lại mỗi request.
"""

import os
import sys
from contextlib import asynccontextmanager

# Đảm bảo import được `src.*` dù chạy từ đâu
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from sqlalchemy import text

from src.data.dataloader import FEATURE_COLS, WINDOW_SIZE
from src.data.dbmanager import get_all, get_engine

MODEL_PATH  = os.path.join(PROJECT_ROOT, "models", "lstm_baseline.h5")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "minmax_scaler.pkl")

# Nơi giữ model/scaler sau khi load lúc startup
state: dict = {"model": None, "scaler": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Chạy 1 lần khi server khởi động: load TensorFlow + model + scaler."""
    try:
        import joblib
        from tensorflow.keras.models import load_model
        state["model"] = load_model(MODEL_PATH, compile=False)
        state["scaler"] = joblib.load(SCALER_PATH)
        print("✅ Đã load model + scaler.")
    except Exception as e:  # thiếu file / lỗi phiên bản TensorFlow
        state["error"] = str(e)
        print(f"❌ Không load được model: {e}")
    yield
    # (không cần dọn dẹp gì khi tắt server)


app = FastAPI(title="AQI Prediction API", lifespan=lifespan)


# Cache dữ liệu trong RAM: chỉ đọc + xử lý lại khi DB thay đổi (có bản ghi mới hoặc bị xóa bản ghi cũ)
# "max_dt" lưu cặp (MIN(dt), MAX(dt))
_cache: dict = {"max_dt": None, "df": None, "json": None}


def _get_df():
    """Trả DataFrame từ cache nếu DB chưa có gì mới, ngược lại đọc lại từ DB."""
    with get_engine().connect() as conn:
        key = tuple(conn.execute(text("SELECT MIN(dt), MAX(dt) FROM air_quality")).one())
    if _cache["df"] is None or key != _cache["max_dt"]:
        _cache.update(max_dt=key, df=get_all(), json=None)
    return _cache["df"]


@app.get("/api/air-quality")
def air_quality():
    """Toàn bộ dữ liệu dạng list of records. `datetime` là chuỗi ISO, giờ UTC.
    Ô thiếu dữ liệu trả về null."""
    df = _get_df()
    if _cache["json"] is None:
        # to_json của pandas nhanh hơn nhiều so với to_dict + FastAPI tự mã hóa; NaN → null
        _cache["json"] = df.reset_index().to_json(
            orient="records", date_format="iso", date_unit="s")
    return Response(content=_cache["json"], media_type="application/json")


@app.get("/api/predict/next-hour")
def predict_next_hour():
    """Dự đoán lớp AQI (1..5) của giờ kế tiếp từ 24 giờ gần nhất trong DB."""
    if state["model"] is None:
        raise HTTPException(503, f"Model chưa sẵn sàng: {state['error']}")

    df = _get_df()
    if df.empty:
        raise HTTPException(503, "Chưa có dữ liệu trong DB.")

    last_valid = df["aqi"].last_valid_index()
    win = df.loc[:last_valid, FEATURE_COLS].tail(WINDOW_SIZE)   # 24 giờ liên tiếp
    if len(win) < WINDOW_SIZE or win.isna().any().any():
        raise HTTPException(422, "24 giờ gần nhất còn thiếu dữ liệu nên không dự đoán được.")

    X = state["scaler"].transform(win).astype("float32")[None, ...]
    probs = state["model"].predict(X, verbose=0)[0]
    return {
        "level": int(np.argmax(probs)) + 1,
        "probs": [float(p) for p in probs],
        "based_on": last_valid.strftime("%Y-%m-%dT%H:%M:%S"),   # giờ cuối cùng (UTC) của cửa sổ 24h
    }


@app.get("/health")
def health():
    """Kiểm tra DB kết nối được không và model đã load chưa."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return {
        "db": db_ok,
        "model": state["model"] is not None,
        "model_error": state["error"],
    }
