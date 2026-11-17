# Nhật ký thực nghiệm — mmxai-lab

Viết tay, lượt mới nhất ở trên. Mỗi lượt đầy đủ (hoặc mỗi kết luận từ nhiều lượt thử) một đoạn. Số lấy từ `metrics.json` /
`REPRODUCTION.md`, không ước lượng. Paper: Project 3 - Truy vấn đa phương thức có giải thích bằng đồ thị (Graph ML).

## Mẫu đoạn

```
## #N · YYYY-MM-DD · <run_id> (`--tag ...`)

- **Mục đích:** tái lập Bảng X dòng Y / kiểm tra giả thuyết Z / ablation ...
- **Cấu hình:** model, dữ liệu, tham số khác mặc định (`--set ...`), compute (Kaggle T4 / Colab), commit vendor.
- **Kết quả:** metric chính (paper: a · mình: b · Δ) ; thời gian/mẫu, VRAM; lỗi parse/crash nếu có.
- **Nhận xét & kết luận:** khớp/không khớp vì sao; khác biệt so với paper; bước tiếp theo.
- **Output:** `runs/<exp>/<run_id>/`
```

## #5 · 2026-11-17 · `t1_qa` — điểm đồ thị phụ thuộc truy vấn (khái niệm của câu ↔ nhãn của ảnh), seed 0

- **Mục đích:** sửa nguyên nhân của #4 (gộp nhãn thành một vector mất thông tin khớp). `graph/qa.py`: GNN chỉ trên đồ thị nhãn,
  m(c,i) = max mềm cos(u_c, H_l) trên nhãn l của ảnh, s_G = trung bình theo khái niệm của caption (`concepts` của Thắng, 100%
  nằm trong bộ 2000 nhãn); score = adapter + w·s_G. Test: no_edge ⇒ s_G = 0; đường đánh giá nhanh = đường huấn luyện.
- **Cấu hình:** GAT, chưa ConceptNet, batch 512. Máy bị lag khi 4 lượt song song → chạy tuần tự `nice`, 2 luồng (~44 phút/lượt).
- **Kết quả test seed 0 (toàn pool R@1 / MRR · gallery 1000 MRR):**
  CLIP 48.31 / 60.40 · 69.79 — khớp nhãn CLIP không học (untrained, không adapter) 48.40 / 60.47 · 69.93 —
  CLIP+Adapter (qa no_edge) 51.13 / 63.26 · 72.33 — **CLIP+Adapter+QA-Graph 52.11 / 64.30 · 73.71** (α 0.25).
  MRR theo nhóm (gallery): adapter direct 73.46 / indirect 72.06 → qa **direct 80.54** / indirect 72.12.
- **Nhận xét:** lần đầu đồ thị thêm được trên adapter: +1.04 MRR toàn pool, +1.4 gallery, +1.0 R@1 (1 seed). Toàn bộ phần
  tăng nằm ở nhóm **direct** (+7.1); indirect không đổi → chưa có bằng chứng lan truyền qua sg_rel giúp câu gián tiếp
  (giống AdaMF-MAT của Khánh). Khớp nhãn không học gần như vô dụng (+0.07) → phần học (u_c, τ, w) là cần thiết.
- **Tiếp theo:** `no_rel` (bị ngắt khi phiên kết thúc) để biết cạnh nhãn–nhãn có góp gì; ConceptNet cho nhóm indirect;
  3 seed; kiểm định ghép cặp với adapter.
- **Output:** `runs/t1_qa/20261117-032408_huy-chau_t1_qa-seed0/`, `runs/t1_qa_ablation_no_edge/20261117-030513_*`

## #4 · 2026-11-17 · kiến trúc `split` (adapter + đồ thị tách rời), seed 0 (`--tag split-seed0`)

- **Mục đích:** sửa lỗi của #3. Nhánh đồ thị chỉ tổng hợp từ láng giềng (ảnh không self-loop), điểm = cos(Pq,Sx) + w·cos(Gq,g_i).
  Test mới: no_edge ⇒ thứ hạng ≡ adapter; GCN/SAGE: g_i không phụ thuộc đặc trưng ảnh. Dữ liệu giả xác nhận kiến trúc dùng
  được đồ thị: MRR CLIP 0.321 · no_edge 0.337 · shuffle 0.334 · **full 0.384** (indirect 0.103 → 0.215).
- **Cấu hình:** như #2 (GAT, chưa ConceptNet), `gnn.arch=split`; 4 lượt song song trên CPU.
- **Kết quả test seed 0, R@1 / MRR (gallery 1000 · toàn pool):**
  CLIP 58.17 / 69.79 · 48.31 / 60.40 —
  **CLIP+Adapter (no_edge)** 60.99 / 72.40 · 51.22 / 63.35 —
  **+Graph (full)** 60.95 / 72.48 · 50.97 / 63.25 —
  +Graph shuffle 60.41 / 72.10 · 50.90 / 63.17 —
  **Graph trên CLIP thô (adapter=false)** 58.87 / 70.37 · 48.64 / 60.83 (α = 0.7).
  MRR direct/indirect: no_edge 73.34/72.17 · full 73.79/72.18 · shuffle 73.30/71.83.
- **Nhận xét:** kiến trúc đã đúng (dữ liệu giả) nhưng trên dữ liệu thật **đồ thị nhãn VG (chưa ConceptNet) không thêm gì
  trên adapter** (chênh ≤ 0.1 MRR, trong nhiễu 1 seed); trên CLIP thô chỉ +0.4 MRR toàn pool — dưới cả baseline khớp tên
  của Khánh (+1.06). Giả thuyết: g_i gộp ~17 nhãn của ảnh thành MỘT vector không phụ thuộc truy vấn → mất thông tin
  "nhãn nào khớp khái niệm nào của câu", thứ mà khớp tên khai thác được; CLIP ảnh vốn đã mã hóa "túi đối tượng".
- **Tiếp theo:** điểm đồ thị phụ thuộc truy vấn (khái niệm của caption `concepts` ↔ nhãn của ảnh qua đường đi trong đồ thị),
  thêm ConceptNet; cùng bộ ablation.
- **Output:** `runs/t1_*/2026111[67]-*split*`

## #3 · 2026-11-16 · t1_ablation_no_edge / t1_ablation_shuffle (`--seed 0 --tag seed0`) — đồ thị CHƯA đóng góp

- **Mục đích:** kiểm tra phần tăng của lượt #2 đến từ cấu trúc đồ thị hay từ lớp chiếu học được.
- **Cấu hình:** giống #2 (GAT, chưa ConceptNet), chỉ đổi `graph.variant`. Chạy song song 2 lượt trên CPU (4 luồng mỗi lượt).
- **Kết quả test seed 0 (R@1 / MRR):** gallery 1000 — CLIP 58.17 / 69.79 · full 60.45 / 72.15 · **no_edge 60.87 / 72.36** ·
  **shuffle 61.15 / 72.48**. Toàn pool — CLIP 48.31 / 60.40 · full 50.94 / 63.19 · no_edge 50.90 / 63.14 · shuffle 51.00 / 63.17.
  α = 0.35 / 0.30 / 0.35. Val MRR(graph) của cả ba đều đỉnh ở epoch 0–1 (≈0.728–0.730).
- **Nhận xét:** bỏ hết cạnh hay xáo cạnh cho cùng mức tăng → **toàn bộ +2.8 MRR là của adapter** (q_proj, skip, và GAT
  self-loop biến đổi chính đặc trưng ảnh = MLP), không phải tri thức đồ thị. Baseline đúng để so phải là "CLIP + adapter"
  (no_edge), không phải CLIP thô. Số của #2 KHÔNG được báo cáo là đóng góp của đồ thị.
- **Tiếp theo:** sửa mô hình để nhánh GNN chỉ nhận tin từ nhãn/khái niệm (ảnh không self-loop trong GNN), tách adapter
  thành thành phần riêng; thêm ConceptNet; chạy lại full vs no_edge vs shuffle.
- **Output:** `runs/t1_ablation_no_edge/20261116-150308_huy_t1_ablation_no_edge-seed0/`,
  `runs/t1_ablation_shuffle/20261116-150308_huy_t1_ablation_shuffle-seed0/`

## #2 · 2026-11-15 · 20261115-082025_huy-chau_t1_graph-seed0-dau-tien — dữ liệu THẬT, 1 seed (chưa phải số báo cáo)

- **Mục đích:** lượt CLIP+Graph đầu tiên trên bundle Nhiệm vụ 1 của Thắng (VG∩COCO, Karpathy split, train 15k ảnh /
  75k caption, val 2126, test 2138), đánh giá bằng `mmxai_eval.Protocol` của Thắng (gallery 1000 theo seed + full pool,
  `fuse` z-score, `tune_alpha` trên val bước 0.05).
- **Kiểm tra trước:** `t1_clip` khớp tuyệt đối baseline của Thắng (R@1 57.36, MRR 69.23 mean 3 gallery; full 48.31 / 60.40);
  `t1_untrained` (GNN khởi tạo đơn vị) cho đúng số CLIP → đường dữ liệu đúng.
- **Cấu hình:** GAT 2 tầng, hidden 256, 4 head; đồ thị 19 264 ảnh + 2 000 nhãn (`object_vocab_train`), 334k cạnh contains,
  98.8k cạnh sg_rel (train, `min_rel_count=1`), CHƯA có ConceptNet. batch 512, lr 1e-3, tau 0.05, patience 4. CPU 4 luồng.
- **Kết quả test (seed 0):** gallery 1000: CLIP R@1 58.17 / MRR 69.79 → **60.45 / 72.15** (+2.3 / +2.4);
  full pool: 48.31 / 60.40 → **50.94 / 63.19** (+2.6 / +2.8). α = 0.35. Nhóm direct / indirect (định nghĩa của Thắng):
  MRR 72.84 / 71.99.
- **Nhận xét:** cải thiện thật trên dữ liệu thật. MRR val (graph) đạt đỉnh ở epoch 1 rồi giảm → quá khớp sớm; còn chỗ dò
  lr/dropout/weight decay. CHƯA biết phần cải thiện đến từ cấu trúc đồ thị hay từ lớp chiếu học được → bắt buộc chạy
  `no_edge` và `shuffle` trên dữ liệu thật. 27 phút/seed trên CPU (≈4.4 phút/epoch).
- **Tiếp theo:** ablation no_edge + shuffle (seed 0), ConceptNet (A3), rồi 3 seed.
- **Output:** `runs/t1_graph/20261115-082025_huy-chau_t1_graph-seed0-dau-tien/`

## #1 · 2026-11-14 · kiểm tra pipeline trên dữ liệu GIẢ (local CPU, 1 seed, chưa phải kết quả báo cáo)

- **Mục đích:** xác nhận pipeline phát hiện được đóng góp của cấu trúc đồ thị trước khi tốn GPU cho VG thật.
- **Cấu hình:** `clip_graph` (GAT 2 tầng), dữ liệu giả 2000 ảnh / 4000 caption / 80 nhãn (`img_noise=0.15, cap_noise=0.1`),
  `train.batch_size=64, patience=8, epochs≤60`, seed 0; biến thể `full`, `no_edge`, `shuffle`.
- **Kết quả (MRR test, all / direct / indirect):** CLIP 0.321 / 0.459 / 0.103 · full 0.399 / 0.509 / 0.223 ·
  no_edge 0.332 / 0.459 / 0.130 · shuffle 0.331 / 0.467 / 0.116. Fidelity (full): xóa cạnh giải thích Δ ≈ 0.10–0.13,
  xóa ngẫu nhiên cùng số cạnh Δ ≈ 0.0001. ~70 s/seed trên CPU 4 luồng.
- **Nhận xét:** cải thiện tập trung ở nhóm gián tiếp (×2 MRR) và biến mất khi bỏ hoặc xáo cạnh → pipeline tách được vai trò
  của cấu trúc, đúng thiết kế dữ liệu giả. Hai lỗi sửa trong lúc làm: (1) huấn luyện cả đồ thị mỗi bước quá chậm → đồ thị con
  theo batch (chính xác, có test); (2) torch 16 luồng chậm ~15× → `cpu_threads: 4`. Kernel Kaggle sửa lỗi `VENDOR["name"]=None`
  và cài `torch_geometric` qua `vendor.yaml: pip`.
- **Tiếp theo:** nhận dữ liệu VG thật (Yến) + features (Thắng) theo `docs/data-contract.md`; `make push --limit 5` trên Kaggle.
- **Output:** `runs/clip_graph/20261114-04*` (local, không commit).

## #0 · 2026-11-14 · khởi tạo

- Dựng harness bằng skill `paper-lab`; `make test` và `make local EXP=smoke` chạy được.
- Kế hoạch: `docs/plan.md`. Bảng cần tái lập: `docs/paper_numbers.yaml`.
