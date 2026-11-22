# report/ — báo cáo IEEE (LaTeX, tiếng Việt)

```bash
make report        # từ gốc repo: vẽ hình (make_figs.py → figs/*.pdf) + XeLaTeX 2 lượt → build/report/report.pdf
```

- Cần: XeLaTeX + gói `texlive-xetex texlive-latex-extra texlive-pictures fonts-texgyre` (Debian/Ubuntu), và `make setup`
  (matplotlib cho hình). Font thân bài: Times New Roman nếu máy có, không thì TeX Gyre Termes (đã khai báo trong `report.tex`).
- `report.tex`: nội dung; chỗ còn thiếu đánh dấu `\todo{...}`. `make_figs.py`: vẽ hình từ số liệu chép tay trong script, nguồn
  `results/tables/*.md` và `results/adamf/vs_qagraph_paired.md` (đổi kết quả thì sửa số trong script). `IEEEtran.cls` đi kèm, không cần cài.
- File trung gian (`.aux`, `.log`) nằm trong `build/report/`, không commit. Quy định nộp: 1 file PDF, ≤ 25 trang.
- Overleaf: tải lên `report.tex`, `IEEEtran.cls`, `figs/`; chọn trình biên dịch XeLaTeX.
- Hình 1 (`figs/fig_overview.tex`) minh họa bằng một truy vấn thật của tập test PSG; số liệu và ảnh sinh bằng `../.venv/bin/python make_example.py` (cần `data/psg_bundle`, `data/psg_raw`, `models/psg_qa_cn_seed0`) → `figs/example.json`, `figs/ex_*.jpg`.
