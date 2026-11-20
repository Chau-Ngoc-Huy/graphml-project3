# report/ — báo cáo IEEE (LaTeX)

- `report.tex`: nội dung (tiếng Việt). Chỗ còn thiếu đánh dấu `[CẦN BỔ SUNG: ...]` (xAI/MMKG của Dương, link W&B, họ tên).
- `make_figs.py`: vẽ `figs/*.pdf` từ số liệu trong JOURNAL (chạy trong `report/`).
- Biên dịch (cần font Times New Roman, XeTeX): `tectonic report.tex` hoặc `xelatex report.tex` (chạy 2 lần).
- Quy định nộp: 1 file PDF, ≤ 25 trang, tên `MSHV1-MSHV2-...pdf`.

Sửa xong: `git pull --rebase`, commit, `git push`. Mỗi người sửa phần của mình trong `report.tex` và báo nhóm để tránh conflict.
