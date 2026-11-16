# Lệnh dùng hằng ngày. Chạy từ thư mục khanh/.
PY ?= .venv/bin/python
KAGGLE ?= .venv/bin/kaggle
DATA ?= ../../data/scene_graphs.json
KERNEL ?= khnhnguyninhquang/adamf-mat-kaggle

.PHONY: help test queries notebooks status

help:                ## liệt kê lệnh
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s $$'\t'

test:                ## pytest (dữ liệu giả, không cần GPU hay AdaMF-MAT)
	$(PY) -m pytest -q

queries:             ## sinh lại bộ 24 truy vấn từ scene_graphs.json (không commit file đó)
	$(PY) src/build_eval_queries.py $(DATA) queries 12 0

notebooks:           ## sinh kaggle/*.ipynb từ src/ (+ ../thang/mmxai_eval.py)
	$(PY) tools/make_notebooks.py

status:              ## trạng thái kernel Kaggle (cần ~/.kaggle/access_token); KERNEL=<user>/<slug>
	$(KAGGLE) kernels status $(KERNEL)
