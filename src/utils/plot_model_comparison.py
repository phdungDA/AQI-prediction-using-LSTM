"""
Vẽ biểu đồ so sánh chỉ số CHUNG (Precision / Recall / F1) của 3 model:
Random Forest -> LSTM -> BiLSTM.

Mỗi ô là một chỉ số, mỗi model có 2 cột:
  - macro avg    : trung bình đều qua 5 lớp AQI (mọi lớp quan trọng như nhau)
  - weighted avg : trung bình có trọng số theo số mẫu từng lớp

Đọc số liệu từ các file eval_metrics_{model}_{split}.json
(chuỗi "classification_report" do evaluate*.py lưu).

Chạy từ thư mục gốc project:
    python src/utils/plot_model_comparison.py                # tập VALIDATION (mặc định)
    python src/utils/plot_model_comparison.py --split test    # tập TEST
    python src/utils/plot_model_comparison.py --split both    # cả val và test (2 ảnh riêng)

Lưu ý: tập test chỉ nên dùng khi đã chốt cấu hình cuối cùng.
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")
OUT_DIR      = os.path.join(PROJECT_ROOT, "reports")

# (tiêu đề ô, vị trí trong bộ (precision, recall, f1))
METRICS = [("Average Precision", 0), ("Average Recall", 1), ("Average F1-Score", 2)]
# (tên dòng trong classification_report, nhãn chú giải, màu)
AVERAGES = [("macro avg", "Macro avg", "#4c72b0"), ("weighted avg", "Weighted avg", "#dd8452")]


def metric_paths(split: str) -> dict:
    """
    Đường dẫn file metrics của từng model, theo thứ tự hiển thị.
    Sửa ở đây nếu file của bạn đặt tên khác.
    """
    return {
        "Random Forest": os.path.join(MODEL_DIR, "randomforest", f"eval_metrics_randomforest_{split}.json"),
        "LSTM":          os.path.join(MODEL_DIR, f"eval_metrics_lstm_{split}.json"),
        "BiLSTM":        os.path.join(MODEL_DIR, f"eval_metrics_bilstm_{split}.json"),
    }


def parse_averages(report: str) -> dict:
    """Đọc dòng 'macro avg' và 'weighted avg' -> {tên dòng: (precision, recall, f1)}."""
    out = {}
    for line in report.splitlines():
        s = line.strip()
        for key, _label, _color in AVERAGES:
            if s.startswith(key):
                p, r, f, _support = map(float, s.split()[-4:])
                out[key] = (p, r, f)
    return out


def main(split: str = "val"):
    paths = metric_paths(split)
    missing = [f"{name}: {p}" for name, p in paths.items() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Thiếu file metrics, hãy chạy evaluate tương ứng trước:\n  " + "\n  ".join(missing)
        )

    data = {}
    for name, path in paths.items():
        with open(path, encoding="utf-8") as f:
            parsed = parse_averages(json.load(f)["classification_report"])
        if len(parsed) != len(AVERAGES):
            raise ValueError(f"Không đọc được dòng macro/weighted avg từ {path}")
        data[name] = parsed

    model_names = list(paths.keys())
    x = np.arange(len(model_names))
    width = 0.34

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (title, idx) in zip(axes, METRICS):
        for j, (key, label, color) in enumerate(AVERAGES):
            vals = [data[m][key][idx] for m in model_names]
            bars = ax.bar(x + (j - 0.5) * width, vals, width, label=label, color=color)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle(f"Evaluation Metrics Comparison ({split})", fontsize=12)
    plt.tight_layout(rect=(0, 0.06, 1, 1))

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"figure_model_comparison_{split}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"✅ Đã lưu: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test", "both"], default="val",
                        help='Tập dữ liệu lấy kết quả: "val" (mặc định), "test" hoặc "both".')
    args = parser.parse_args()
    for s in (["val", "test"] if args.split == "both" else [args.split]):
        main(split=s)
