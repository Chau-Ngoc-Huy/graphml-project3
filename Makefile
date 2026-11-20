# Mọi lệnh nhận EXP=<tên thí nghiệm> (mặc định smoke) và ARGS="..." chuyển cho python -m mmxai.run.
# Cấu hình riêng mỗi người: local.mk (không commit): LAB_USER, KAGGLE_USER, DATASET.
-include local.mk

PY ?= .venv/bin/python
KAGGLE ?= .venv/bin/kaggle
EXP ?= smoke
POLL ?= 30
PROJECT ?= mmxai-lab
export PYTHONPATH := src
export LAB_USER KAGGLE_USER DATASET PROJECT
KERNEL = $(shell $(PY) -c "import sys; sys.path.insert(0,'tools'); from make_kernel import kernel_id; print(kernel_id('$(EXP)'))")

.PHONY: table ablations help test local dry push kernel status wait logs pull queue summary compare tex vendor vendor-update colab check import ui

help:                ## liệt kê lệnh
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s $$'\t'

check:               ## kiểm tra môi trường (python, git, gh, kaggle, GPU)
	@bash tools/check_env.sh

test:                ## pytest toàn bộ
	$(PY) -m pytest -q

local:               ## chạy thí nghiệm trên máy: make local EXP=smoke ARGS="--limit 4"
	$(PY) -m mmxai.run --exp $(EXP) $(ARGS)

dry:                 ## in cấu hình đã hợp nhất của EXP
	$(PY) -m mmxai.run --exp $(EXP) --dry-run $(ARGS)

kernel:              ## chỉ sinh build/kaggle/$(EXP)/run.py (không đẩy)
	$(PY) tools/make_kernel.py --exp $(EXP) $(KERNEL_ARGS) -- $(ARGS)

# make push EXP=main_table ARGS="--limit 4 --tag thu"; từ chối nếu kernel còn đang chạy.
push:                ## sinh kernel + đẩy lên Kaggle
	@if $(KAGGLE) kernels status $(KERNEL) 2>/dev/null | grep -qiE "running|queued"; then \
		echo "Kernel $(KERNEL) đang chạy, chờ xong (make wait EXP=$(EXP)) rồi push lại"; exit 1; fi
	$(PY) tools/make_kernel.py --exp $(EXP) $(KERNEL_ARGS) -- $(ARGS)
	$(KAGGLE) kernels push -p build/kaggle/$(EXP)

status:              ## trạng thái kernel của EXP
	$(KAGGLE) kernels status $(KERNEL)

wait:                ## poll đến khi kernel xong
	@while $(KAGGLE) kernels status $(KERNEL) | tee /dev/stderr | grep -qiE "running|queued"; do sleep $(POLL); done

logs:                ## log kernel (đang chạy hoặc đã xong)
	$(KAGGLE) kernels logs $(KERNEL)

pull:                ## tải kết quả về runs/$(EXP)/<run_id>/ và chép metric sang results/
	$(PY) tools/pull_kernel.py --exp $(EXP)

queue:               ## chạy tuần tự nhiều thí nghiệm: make queue EXPS="a b c" ARGS="--tag v1"
	tools/run_queue.sh "$(EXPS)" "$(ARGS)"

import:              ## nhập lượt chạy từ Colab/máy khác: make import SRC=~/Downloads/runs.zip
	$(PY) tools/import_runs.py $(SRC)

summary:             ## sinh lại results/<exp>/SUMMARY.md + index.csv
	$(PY) -m mmxai.core.results $(if $(filter command line,$(origin EXP)),--exp $(EXP))

compare:             ## so với số trong paper → results/REPRODUCTION.md
	$(PY) tools/compare.py $(ARGS)

tex:                 ## xuất bảng LaTeX: make tex ARGS="--table table2 --out ../report/tables/tab_table2.tex"
	$(PY) tools/export_tex.py $(ARGS)

vendor:              ## lấy repo gốc về vendor/ (submodule) đúng commit đã ghim
	git submodule update --init --recursive

vendor-update:       ## đổi commit ghim của repo gốc: make vendor-update REF=<sha|tag|branch>
	$(PY) tools/vendor_pin.py $(REF)

colab:               ## in hướng dẫn chạy trên Colab (dự phòng khi hết GPU Kaggle)
	@echo "1. Mở colab/run.ipynb trên Google Colab (File → Upload notebook), chọn runtime GPU."
	@echo "2. Điền EXP/ARGS ở ô đầu, chạy hết; kết quả nằm ở Drive: paper-lab/$(PROJECT)/runs/."
	@echo "3. Tải runs.zip về rồi: make import SRC=~/Downloads/runs.zip"

table:               ## bảng kết quả (Markdown + LaTeX) từ lượt đầy đủ mới nhất: results/tables/
	$(PY) tools/report_table.py $(ARGS)

ablations:           ## chạy mọi ablation tuần tự trên Kaggle: make ablations ARGS="--set data=mmxai-vg"
	tools/run_queue.sh "ablation_no_rel ablation_no_concept ablation_no_edge ablation_shuffle ablation_random" "$(ARGS)"

ui:                  ## giao diện thử mô hình / xem dữ liệu / ablation: http://127.0.0.1:8765
	OMP_NUM_THREADS=2 nice -n 10 $(PY) -m uvicorn apps.lab.server:app --host 127.0.0.1 --port 8765
