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

.PHONY: queries notebooks sync table ablations help test local dry push kernel status wait logs pull queue summary compare tex vendor vendor-update colab check import ui

help:                ## liệt kê lệnh
	@grep -hE '^[a-z0-9-]+:.*##' Makefile | awk 'BEGIN{FS=":.*## *"}{printf "  %-16s %s\n", $$1, $$2}'

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

# --- AdaMF-MAT / bộ truy vấn 
SG ?= data/scene_graphs.json
queries:             ## sinh lại bộ 24 truy vấn từ scene_graphs.json của VG (không commit file đó): SG=<đường dẫn>
	$(PY) -m mmxai.adamf.build_eval_queries $(SG) queries 12 0

notebooks:           ## nhúng lại src/mmxai/adamf + src/mmxai_eval.py vào notebooks/kaggle/adamf/*.ipynb
	$(PY) tools/make_notebooks.py

sync:                ## đồng bộ thay đổi của nhóm từ origin/main sang nhánh final (DRY=1: chỉ liệt kê)
	$(PY) tools/sync_main.py $(if $(DRY),--dry-run)

ui:                  ## giao diện thử mô hình / xem dữ liệu / ablation: http://127.0.0.1:8765
	OMP_NUM_THREADS=2 nice -n 10 $(PY) -m uvicorn apps.lab.server:app --host 127.0.0.1 --port 8765

# =====================================================================================================================
# PIPELINE ĐẦY ĐỦ (docs/PIPELINE.md). DS=vg|psg. Bước GPU chạy trên Kaggle bằng kaggle CLI, không thao tác tay trên web.
#   make setup → make fetch → make data DS=.. → make extras DS=.. → make publish DS=.. → make train DS=.. → make adamf DS=..
#   → make tables → make xai DS=.. → make web / make ui / make report
# Đường tắt (dùng bundle + checkpoint đã dựng sẵn trên Kaggle): make setup fetch-prebuilt tables xai web
# =====================================================================================================================
DS ?= psg
MODEL_vg ?= models/vg_t1_qa_cn_seed0
MODEL_psg ?= models/psg_qa_cn_seed0
BUNDLE_vg = data/vg_bundle
BUNDLE_psg = data/psg_bundle
.PHONY: setup fetch fetch-prebuilt data data-vg data-psg extras publish train adamf tables xai web report all-psg all-vg

setup:               ## tạo .venv (Python ≥3.10, torch CPU) + cài gói mmxai đầy đủ phần phụ
	test -d .venv || python3 -m venv .venv
	.venv/bin/python -m pip install -q --upgrade pip
	.venv/bin/python -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
	.venv/bin/python -m pip install -q -e ".[dev,clip,ui,viz]"
	@echo "Kaggle CLI: đặt token ở ~/.kaggle/access_token (hoặc kaggle.json); KAGGLE_USER trong local.mk"

fetch:               ## bước 0: tải dữ liệu thô công khai (caption Karpathy, ConceptNet, PSG)
	$(PY) tools/fetch_data.py captions conceptnet psg

fetch-prebuilt:      ## đường tắt: tải bundle VG/PSG + 4 checkpoint đã dựng sẵn (Kaggle, riêng tư)
	$(PY) tools/fetch_data.py bundles models

data: data-$(DS)     ## bước 1: dựng bundle cho DS
data-vg:             ## bước 1 VG: notebook giao thức đánh giá trên Kaggle GPU → data/vg_bundle  (SMOKE=1: lượt thử)
	$(PY) tools/kaggle_job.py run vg-data --install $(if $(SMOKE),--smoke)
data-psg:            ## bước 1 PSG (local CPU ~75 phút): PSG∩COCO → data/psg_bundle + đặc trưng CLIP
	$(PY) -I tools/psg_to_bundle.py --raw data/psg_raw --coco data/raw/dataset_coco.json \
	    --conceptnet data/conceptnet/hf_en_simple/train.parquet --out $(BUNDLE_psg)
	OMP_NUM_THREADS=2 nice -n 19 $(PY) -I tools/psg_extract_clip.py --bundle $(BUNDLE_psg) --raw data/psg_raw --threads 2

extras:              ## bước 2: vector nhãn + cạnh ConceptNet (graph_extra, graph_extra_cn) cho DS
	tools/build_extras.sh $(DS)

publish:             ## bước 2b: đưa bundle DS lên Kaggle Dataset riêng tư (để kernel huấn luyện đọc)
	tools/publish_bundle.sh $(DS)

train:               ## bước 3: mọi cấu hình QA-Graph × 3 seed (vg: Kaggle 2 worker; psg: local) — TARGET=kaggle|local
	tools/run_suite.sh $(DS) $(TARGET)

adamf:               ## bước 4: AdaMF-MAT + ablation trên Kaggle GPU → results/adamf_$(DS)/  (SMOKE=1: lượt thử)
	$(PY) tools/kaggle_job.py run adamf --dataset $(DS) --install $(if $(SMOKE),--smoke)

tables:              ## bước 5: bảng chính thức + kiểm định ghép cặp → results/tables/, results/*_paired, psg_vs_adamf
	$(PY) tools/vg_table.py
	$(PY) tools/psg_table.py
	test ! -f results/adamf_psg/results_PSG_full.json || $(PY) tools/psg_vs_adamf.py --adamf results/adamf_psg
	test ! -f data/vg_bundle/data/images.jsonl || $(PY) tools/paired_vs_qagraph.py --bundle data/vg_bundle

xai:                 ## bước 6: giải thích + fidelity/sparsity/validity (khung xAI (mmxai.xai)) trên mô hình thật, N truy vấn
	OMP_NUM_THREADS=2 nice -n 10 $(PY) tools/xai_bridge_demo.py --model $(MODEL_$(DS)) --bundle $(BUNDLE_$(DS)) --n $(or $(N),10)

web:                 ## bước 7: MMXAI Observatory — sinh dữ liệu từ results/ + build (apps/observatory/dist)
	cd apps/observatory && npm ci --no-audit --no-fund && npm run data && npm run build

report:              ## bước 8: hình + PDF báo cáo (cần tectonic hoặc xelatex)
	cd report && ../$(PY) make_figs.py && (tectonic report.tex || (xelatex report.tex && xelatex report.tex))

all-psg:             ## toàn bộ PSG: fetch → data → extras → publish → train (local) → adamf (Kaggle) → tables → xai
	$(MAKE) fetch && $(MAKE) data DS=psg && $(MAKE) extras DS=psg && $(MAKE) publish DS=psg && \
	$(MAKE) train DS=psg && $(MAKE) adamf DS=psg && $(MAKE) tables && $(MAKE) xai DS=psg

all-vg:              ## toàn bộ VG: data (Kaggle) → extras → publish → train (Kaggle) → adamf (Kaggle) → tables → xai
	$(MAKE) fetch && $(MAKE) data DS=vg && $(MAKE) extras DS=vg && $(MAKE) publish DS=vg && \
	$(MAKE) train DS=vg && $(MAKE) adamf DS=vg && $(MAKE) tables && $(MAKE) xai DS=vg
