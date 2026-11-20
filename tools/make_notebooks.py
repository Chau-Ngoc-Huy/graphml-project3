"""Nhúng lại mã nguồn hiện tại vào các ô `%%writefile` của notebook Kaggle (notebook nhúng nguyên mã nguồn để chạy độc lập).

Sửa mã trong src/adamf/ (hoặc src/mmxai_eval.py), rồi `make notebooks`. Phần cấu hình và các stage (ô khác) sửa trực tiếp
trong notebook. Chỉ đổi nội dung các ô `%%writefile /kaggle/working/src/<tên>.py`; in ra file nào đổi.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {"mmxai_eval.py": ROOT / "src" / "mmxai_eval.py"}
SOURCES.update({p.name: p for p in (ROOT / "src" / "adamf").glob("*.py")})
PAT = re.compile(r"^%%writefile /kaggle/working/src/([\w.]+)\n")

for nb_path in sorted((ROOT / "notebooks" / "kaggle" / "adamf").glob("*.ipynb")):  # exploratory/ là bản lưu các lượt đã chạy: không đụng
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    changed = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        m = PAT.match(src)
        if not m or m.group(1) not in SOURCES:
            continue
        new = m.group(0) + SOURCES[m.group(1)].read_text(encoding="utf-8")
        if new != src:
            cell["source"] = new.splitlines(keepends=True)
            changed.append(m.group(1))
    if changed:
        nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{nb_path.relative_to(ROOT)}: " + (", ".join(changed) if changed else "đã khớp src/"))
