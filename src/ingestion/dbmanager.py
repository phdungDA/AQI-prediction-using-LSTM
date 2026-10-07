import csv
import os
import time

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_batch

# ── Kết nối PostgreSQL ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Thiếu DATABASE_URL trong .env (vd: postgresql://user:pass@host:5432/dbname)")

# Khớp với api_client.py và src/data/dataloader.py
RAW_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "air_quality_3years.csv")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS air_quality (
    dt        BIGINT PRIMARY KEY,
    datetime  TIMESTAMP,
    aqi       INTEGER,
    co        FLOAT,
    no        FLOAT,
    no2       FLOAT,
    o3        FLOAT,
    so2       FLOAT,
    pm2_5     FLOAT,
    pm10      FLOAT,
    nh3       FLOAT
);
"""

INSERT_SQL = """
INSERT INTO air_quality (dt, datetime, aqi, co, no, no2, o3, so2, pm2_5, pm10, nh3)
VALUES (%(dt)s, %(datetime)s, %(aqi)s, %(co)s, %(no)s, %(no2)s, %(o3)s, %(so2)s, %(pm2_5)s, %(pm10)s, %(nh3)s)
ON CONFLICT (dt) DO NOTHING;
"""


def get_connection():
    """Tạo kết nối đến PostgreSQL bằng DATABASE_URL đọc từ .env."""
    return psycopg2.connect(DATABASE_URL)


def init_db():
    """Tạo bảng air_quality nếu chưa có (an toàn khi bảng đã tồn tại sẵn)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
        conn.commit()
    print("✅ Bảng air_quality đã sẵn sàng.")


def _to_int(value: str) -> int:
    """Ép về int, chịu được dạng lẻ kiểu "2.0", "1500.00" (CSV có thể lưu số nguyên
    dưới dạng float)."""
    return int(round(float(value)))


def _to_float(value: str) -> float | None:
    """Ép về float, trả None nếu ô trống/NA thay vì crash (ON CONFLICT DO NOTHING
    vẫn insert được NULL cho các cột không phải khóa chính)."""
    if value is None or value.strip() == "" or value.strip().upper() in ("NA", "NAN", "NULL"):
        return None
    return float(value)


def load_from_csv(path: str = RAW_CSV_PATH) -> list[dict]:
    """Đọc dữ liệu từ file CSV raw. Ép kiểu chịu được số lẻ ("2.0" cho cột int)
    và ô trống (NA) cho các cột float."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "dt"      : _to_int(row["dt"]),
                "datetime": row["datetime"],
                "aqi"     : _to_int(row["aqi"]),
                "co"      : _to_float(row["co"]),
                "no"      : _to_float(row["no"]),
                "no2"     : _to_float(row["no2"]),
                "o3"      : _to_float(row["o3"]),
                "so2"     : _to_float(row["so2"]),
                "pm2_5"   : _to_float(row["pm2_5"]),
                "pm10"    : _to_float(row["pm10"]),
                "nh3"     : _to_float(row["nh3"]),
            })
    return records


def insert_records(records: list[dict]) -> None:
    """Đẩy toàn bộ records vào PostgreSQL, bỏ qua nếu đã tồn tại (ON CONFLICT DO NOTHING)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_batch(cur, INSERT_SQL, records, page_size=500)
        conn.commit()
    print(f"✅ Đã insert {len(records)} bản ghi vào PostgreSQL.")


def count_rows() -> int:
    """Đếm số dòng hiện có trong bảng air_quality (dùng để verify sau khi insert)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM air_quality;")
            return cur.fetchone()[0]


def get_latest_dt() -> int | None:
    """Trả về timestamp (dt) mới nhất trong bảng, hoặc None nếu bảng trống.
    Dùng để biết cần lấy dữ liệu mới từ thời điểm nào."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(dt) FROM air_quality;")
            return cur.fetchone()[0]


RETENTION_DAYS = 5 * 365   # chỉ giữ dữ liệu trong ~5 năm gần nhất


def delete_older_than(days: int = RETENTION_DAYS) -> int:
    """Xóa các bản ghi cũ hơn `days` ngày (tính từ bây giờ, theo UTC).
    Trả về số dòng đã xóa."""
    cutoff_ts = int(time.time()) - days * 24 * 3600
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM air_quality WHERE dt < %s;", (cutoff_ts,))
            deleted = cur.rowcount
        conn.commit()
    return deleted


# ── Chạy trực tiếp ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()

    print(f"Đang đọc data từ {RAW_CSV_PATH} ...")
    records = load_from_csv()
    print(f"✅ Đọc được {len(records)} bản ghi.")

    insert_records(records)

    total = count_rows()
    print(f"📊 Tổng số dòng hiện có trong bảng air_quality: {total}")
