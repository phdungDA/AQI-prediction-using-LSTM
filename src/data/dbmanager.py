"""
Đọc dữ liệu từ PostgreSQL cho backend (FastAPI).

get_all() trả về DataFrame đã:
  - đưa về lưới 1 giờ liên tục (giờ bị thiếu → NaN)
  - nội suy tuyến tính gap ngắn (<= MAX_GAP_HOURS) cho các chất ô nhiễm
  - cột AQI KHÔNG nội suy

Logic này chuyển từ app/Interface/common.py (load_data) sang backend,
giữ giống src/data/dataloader.py để dữ liệu hiển thị và dữ liệu train cùng phân phối.
"""

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.data.dataloader import FEATURE_COLS, MAX_GAP_HOURS, _long_gap_mask

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

DATABASE_URL = os.getenv("DATABASE_URL")

# Thứ tự cột trả về, khớp format load_data() cũ
COLUMNS = ["aqi"] + FEATURE_COLS

_engine: Engine | None = None


def get_engine() -> Engine:
    """Tạo engine SQLAlchemy 1 lần rồi dùng lại (engine tự quản lý connection pool)."""
    global _engine
    if _engine is None:
        if not DATABASE_URL:
            raise RuntimeError("Thiếu DATABASE_URL trong .env")
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    return _engine


def get_all() -> pd.DataFrame:
    """
    Đọc toàn bộ bảng air_quality, trả DataFrame:
      index = datetime (lưới 1 giờ liên tục, UTC)
      cột   = [aqi, co, no, no2, o3, so2, pm2_5, pm10, nh3]
    """
    query = text(f"SELECT datetime, {', '.join(COLUMNS)} FROM air_quality ORDER BY datetime")
    with get_engine().connect() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], name="datetime"))

    df["datetime"] = pd.to_datetime(df["datetime"]).dt.floor("60min")
    df = (df.drop_duplicates(subset="datetime", keep="last")
            .set_index("datetime").sort_index())

    full = pd.date_range(df.index.min(), df.index.max(), freq="h", name="datetime")
    df = df.reindex(full)

    for col in FEATURE_COLS:
        is_na = df[col].isna()
        filled = df[col].interpolate(method="linear", limit_area="inside")
        filled[_long_gap_mask(is_na, MAX_GAP_HOURS)] = float("nan")
        df[col] = filled

    return df[COLUMNS]


# ── Chạy trực tiếp để kiểm tra (chạy từ thư mục gốc: python -m src.data.dbmanager) ──
if __name__ == "__main__":
    data = get_all()
    print(data.shape)
    print(data.head())
    print(data.tail())
