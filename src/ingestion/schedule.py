"""
Tự động lấy dữ liệu MỚI từ OpenWeatherMap và ghi vào PostgreSQL mỗi 30 phút.

Mỗi lần chạy:
  1. Hỏi DB: bản ghi mới nhất đang có là lúc nào (MAX(dt))
  2. Gọi API lấy dữ liệu từ sau mốc đó → hiện tại
  3. Insert vào DB (trùng dt thì bỏ qua nhờ ON CONFLICT DO NOTHING)

Chạy từ thư mục gốc project (để chạy liên tục, giữ cửa sổ terminal mở):
    python src/ingestion/schedule.py
Chỉ chạy 1 lần rồi thoát (để test):
    python src/ingestion/schedule.py --once
"""

import time
import argparse
from datetime import datetime, timezone

from apscheduler.schedulers.blocking import BlockingScheduler

# Chạy dạng script nên import trực tiếp các file cùng thư mục
from api_client import fetch_air_quality
from dbmanager import init_db, insert_records, get_latest_dt, delete_older_than

INTERVAL_MINUTES = 30
FIRST_RUN_LOOKBACK = 24 * 3600   # DB trống → chỉ lấy 24h gần nhất


def fetch_new_data() -> None:
    """Lấy dữ liệu mới từ API và ghi vào DB. Có bắt lỗi để job không làm sập scheduler."""
    now = datetime.now(timezone.utc)
    print(f"\n[{now:%Y-%m-%d %H:%M:%S} UTC] Bắt đầu lấy dữ liệu mới...")

    try:
        end_ts = int(time.time())
        latest = get_latest_dt()
        start_ts = (latest + 1) if latest else end_ts - FIRST_RUN_LOOKBACK

        if start_ts >= end_ts:
            print("   Chưa có dữ liệu mới, bỏ qua.")
            return

        records = fetch_air_quality(start_ts, end_ts)
        if not records:
            print("   API chưa trả về bản ghi mới (dữ liệu cập nhật theo giờ).")
            return

        insert_records(records)

    except Exception as e:
        # Lỗi mạng / API / DB: ghi log rồi đợi lần chạy sau (30p) thử lại
        print(f"   ❌ Lỗi khi lấy dữ liệu mới: {e}")


def cleanup_old_data() -> None:
    """Xóa dữ liệu cũ hơn 5 năm khỏi DB."""
    try:
        deleted = delete_older_than()
        if deleted:
            print(f"   🧹 Đã xóa {deleted} bản ghi cũ hơn 5 năm.")
    except Exception as e:
        print(f"   ❌ Lỗi khi xóa dữ liệu cũ: {e}")


def run_job() -> None:
    """1 lần chạy của job: lấy dữ liệu mới, rồi dọn dữ liệu quá cũ."""
    fetch_new_data()
    cleanup_old_data()


def main() -> None:
    parser = argparse.ArgumentParser(description="Lấy dữ liệu AQI mới mỗi 30 phút")
    parser.add_argument("--once", action="store_true", help="Chỉ chạy 1 lần rồi thoát")
    args = parser.parse_args()

    init_db()          # đảm bảo bảng tồn tại
    run_job()          # chạy ngay 1 lần khi khởi động, không đợi 30p đầu tiên

    if args.once:
        return

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        run_job,
        "interval",
        minutes=INTERVAL_MINUTES,
        max_instances=1,    # job trước chưa xong thì không chạy chồng
        coalesce=True,      # lỡ nhiều lần (máy ngủ) thì chỉ chạy bù 1 lần
    )
    print(f"⏰ Scheduler đã chạy: cập nhật mỗi {INTERVAL_MINUTES} phút. Nhấn Ctrl+C để dừng.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Đã dừng scheduler.")


if __name__ == "__main__":
    main()
