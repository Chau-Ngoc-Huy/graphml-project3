"""MẪU adapter: chép thành experiments/<tên>.py, sửa 3 chỗ đánh dấu TODO. Một adapter = một ô/dòng trong bảng của paper.

Nguyên tắc:
- Không sửa code trong vendor/. Mọi khác biệt so với bản gốc (tham số, patch) nằm ở đây hoặc ở configs/experiments/<tên>.yaml
  để người đọc JOURNAL biết chính xác ta đã chạy gì.
- Gọi CLI gốc bằng vendor.run_cmd (log vào runs/<run_id>/vendor.log) hoặc import trực tiếp (vendor.add_to_path).
- Trả về metrics dưới đúng tên khóa dùng trong docs/paper_numbers.yaml để tools/compare.py so được với số trong paper.
- Tôn trọng ctx.limit: lượt thử phải chạy trong vài phút (bớt mẫu, bớt bước, model nhỏ).
"""

from __future__ import annotations

from ..core import vendor


def run(ctx) -> dict:
    vdir = vendor.vendor_dir()
    cfg = ctx.cfg
    out_dir = ctx.run_dir / "vendor_out"
    out_dir.mkdir(exist_ok=True)

    # TODO 1: lệnh gốc (từ docs/repo-audit.md, mục "Bảng → lệnh"). Tham số lấy từ cfg để config là nguồn sự thật.
    cmd = [
        "python", "main.py",
        "--model", cfg["model"]["name"],
        "--dataset", cfg["data"]["name"],
        "--seed", str(ctx.seed),
        "--output_dir", str(out_dir),
    ]
    if ctx.limit:  # TODO 2: cách thu nhỏ của repo này (max_samples, max_steps, subset...)
        cmd += ["--max_samples", str(ctx.limit)]

    res = vendor.run_cmd(cmd, cwd=vdir, log_file=ctx.log_path())

    # TODO 3: lấy số: file kết quả của repo (ưu tiên) hoặc regex trên log.
    results_file = out_dir / "results.json"
    if results_file.exists():
        raw = vendor.read_json(results_file)
        metrics = {"accuracy": float(raw["acc"])}
    else:
        metrics = vendor.grep_metrics(res.stdout, {"accuracy": r"accuracy[:=]\s*([\d.]+)"})
    n = ctx.limit or cfg["data"].get("n")
    return {"metrics": metrics, "n": n, "extra": {"cmd": cmd}}
