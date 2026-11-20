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

## #11 · 2026-11-20 · AdaMF-MAT trên PSG + ablation, Kaggle T4 (`huychau/mmxai-adamf-psg` v1)

- **Mục đích:** phương pháp cấp độ 2 (AdaMF-MAT) trên tập công khai thứ hai, cùng giao thức PSG với #10; thêm ablation của nó.
- **Cấu hình:** notebook nguyên vẹn (`results/adamf_psg/adamf_psg.ipynb`), chỉ đổi input = dataset riêng tư
  `huychau/mmxai-psg-bundle`, tên MMKG `PSG_*`, `RUN_ABLATIONS=True`. MMKG: 133 nhãn PSG + 9.186 ảnh (6.000 train + val/test),
  69.445 bộ ba (65.131 contains + 4.314 quan hệ khái niệm). 28 s/epoch → 300 epoch (trần), 1 seed huấn luyện; full ∥ shuffle trên
  2 GPU rồi drop; tổng ≈ 5 giờ GPU. Lưu ý: từ vựng của AdaMF-MAT chỉ gồm nhãn ảnh, nên khái niệm mở rộng (woman…) bị bỏ qua.
- **Kết quả (MRR %, toàn pool 2186 ảnh, `results/psg_vs_adamf.md`):** CLIP 59.59 · +AdaMF-MAT 60.30 (+0.71 [0.47, 0.95] vs CLIP)
  · khớp tên 60.52 · Adapter 61.40 → +AdaMF-MAT 62.22 (+0.82 vs adapter) ≈ Adapter+khớp tên 62.27 (Δ +0.05 [−0.13, 0.22]).
  Ablation AdaMF-MAT trên nền adapter: xáo cạnh 62.20, bỏ quan hệ 62.18 (Δ −0.02 / −0.04, CI chứa 0).
  QA-Graph 64.29 → hơn Adapter+AdaMF-MAT **+2.07 [1.66, 2.48]** (indirect +2.19, direct +1.29).
  KGC trên quan hệ khái niệm giữ lại: MRR 0.133 (full) / 0.090 (shuffle) / 0.004 (drop) — có học quan hệ, nhưng không giúp tìm ảnh.
- **Nhận xét & kết luận:** lặp lại đúng kết luận trên VG: lợi ích của AdaMF-MAT = thông tin hiện diện đối tượng
  (bằng baseline khớp tên); xáo/bỏ quan hệ không đổi kết quả → phần quan hệ không đóng góp vào xếp hạng. QA-Graph vẫn hơn rõ.
- **Output:** `results/adamf_psg/` (report, results_PSG_*.json có hạng từng truy vấn, notebook), `tools/psg_vs_adamf.py`.

## #10 · 2026-11-20 · DATASET THỨ HAI: PSG — 8 cấu hình × 3 seed, máy local (tag `s0` + `s12`)

- **Mục đích:** kiểm tra kết luận trên VG ở bộ dữ liệu công khai thứ hai cùng cấu trúc VG (scene graph + caption COCO).
- **Dữ liệu:** PSG (ECCV 2022) ∩ caption Karpathy → `data/psg_bundle` (`tools/psg_to_bundle.py`): 18.186 ảnh
  (train 15.000 lấy mẫu seed 0 / val 1.000 / test 2.186), 90.974 caption; 133 nhãn ảnh (lớp panoptic đã chuẩn hoá) +
  674 khái niệm mở rộng (ConceptNet ∩ danh từ trong caption train, ≥20 lần) → truy vấn test indirect 9.645 / direct 1.199 /
  no_concept 89. ConceptNet: 3.743 cạnh. Bỏ 1 ảnh trùng train/val của bản HF. Đặc trưng CLIP: `tools/psg_extract_clip.py`.
- **Cấu hình:** giống VG (GAT, top-100, fuse α tune trên val) nhưng `graph.rel_merge: undirected` (sửa lỗi cạnh sg_rel
  bị nhân đôi khi cặp xuất hiện cả hai chiều; VG giữ `directed` để tái lập #9). CPU nice 19, 2 luồng,
  `tools/run_psg_local.sh` (≈4–16 phút/seed), bảng `tools/psg_table.py` → `results/tables/psg_official.md`.
- **Kết quả (MRR %, toàn pool 2186 ảnh, mean ± std 3 seed · ΔMRR vs Adapter seed 0 bootstrap CI95):**
  CLIP 59.59 · Adapter 62.67 ± 0.03 · QA-Graph contains+sg_rel 64.17 ± 0.09 (+1.58 [1.29, 1.89]) ·
  chỉ contains 64.35 ± 0.08 (+1.59) · +CN 64.41 ± 0.12 (+1.56) · contains+CN 64.21 ± 0.08 (+1.54) ·
  shuffle 62.12 ± 0.08 (−0.50 [−0.78, −0.23]) · CN shuffle 62.17 ± 0.13 (−0.70 [−0.98, −0.42]).
  Direct / indirect MRR (gallery 1K): Adapter 74.6 / 72.0 → QA-Graph 77.2–77.9 / 73.3–73.4.
- **Nhận xét & kết luận:** tái lập gần như y hệt VG: (1) đồ thị thật +1.5–1.6 MRR so với Adapter, CI không chứa 0;
  (2) xáo cạnh làm *tệ hơn* Adapter → lợi ích đến từ cấu trúc contains, không phải thêm tham số; (3) sg_rel và ConceptNet
  không thêm gì đáng kể so với chỉ contains (khác nhau < 0.25, trong nhiễu seed) — kể cả trên PSG nơi 88% truy vấn là
  indirect và từ vựng có 674 khái niệm chỉ nối qua ConceptNet. Nhóm direct hưởng lợi nhiều hơn indirect (≈+3 vs +1.4).
- **Output:** `runs/psg_*/*-s0*`, `runs/psg_*/*-s12*` (model seed 0 lưu để UI dùng), `logs/psg_*.log`.

## #9 · 2026-11-18 · KẾT QUẢ CHÍNH THỨC — 11 cấu hình × 3 seed trên Kaggle T4 (tag `kaggle-3seed`)

- **Cách chạy:** dataset Kaggle private `huychau/mmxai-vg-bundle` (bundle + `graph_extra` + `graph_extra_cn`), mỗi cấu hình
  một kernel chạy seed 0,1,2 (seed = gallery + khởi tạo). Kaggle giới hạn 2 phiên GPU → `tools/run_worker.sh` (2 luồng, thử lại
  push). 4–12 phút/kernel trên T4 (CPU máy: 20–120 phút/seed). Bảng đầy đủ: `results/tables/vg_official.md` (`tools/vg_table.py`).
- **Kết quả (MRR %, toàn pool 2138 ảnh test, mean ± std 3 seed · ΔMRR vs CLIP+Adapter seed 0 bootstrap CI95):**
  CLIP 60.40 · CLIP+Adapter 63.14 ± 0.02 ·
  QA-Graph contains+sg_rel 64.76 ± 0.12 (+1.76 [1.38, 2.16]) · chỉ contains 64.65 ± 0.23 (+1.50) ·
  +CN 64.76 ± 0.26 (+1.75) · contains+CN 64.76 ± 0.04 (+1.65) ·
  GNN (contains+sg_rel+CN): GCN 64.73 · GAT 64.76 · **SAGE 64.89 ± 0.09 (+1.85)**; (contains+CN): GCN 64.56 · GAT 64.76 ·
  **SAGE 64.89 ± 0.15 (+1.93 [1.49, 2.32])** · **shuffle 62.67 ± 0.06 (−0.41 [−0.71, −0.11])**.
  Gallery 1000 MRR: CLIP 69.23 · Adapter 71.80 · QA-Graph 73.4–73.6. Direct / indirect MRR: Adapter 71.90 / 71.80 →
  QA-Graph 78.2–79.3 / 72.1–72.5.
- **Kết luận:**
  1. Đồ thị có đóng góp THẬT trên adapter: +1.4–1.9 MRR toàn pool, CI95 không chứa 0 ở mọi biến thể; so CLIP +4.3–4.5.
  2. Xáo cạnh làm điểm TỤT DƯỚI cả adapter (−0.41, có ý nghĩa) → mô hình dùng đúng nội dung cạnh, không phải số tham số.
  3. Phần tăng chủ yếu ở câu direct (+6.4–7.4); indirect chỉ +0.3–0.7 (~1 std) → lan truyền qua sg_rel/ConceptNet chưa
     giải được câu gián tiếp; khác biệt giữa có/không sg_rel, có/không CN ≤ 0.2 MRR, trong nhiễu.
  4. GNN: SAGE ≥ GAT ≥ GCN nhưng chênh ≤ 0.3 MRR (~1–2 std).
- **Đề xuất cấu hình chính:** contains + ConceptNet · SAGE (`vg_qa_cn_no_sg_sage`): cao nhất (+1.93), đơn giản hơn (không sg_rel).
- **Tiếp theo:** giải thích + fidelity cho cấu hình chính; nối mô hình với khung xAI; viết báo cáo.

## #8 · 2026-11-17 · `vg_qa --set extra=data/vg_bundle/graph_extra_cn` seed 0 — có ConceptNet

- **Cấu hình:** như #5 + ConceptNet (`graph_extra_cn`: 3000 nút = 2000 nhãn + 1000 khái niệm, 32.8k cạnh concept hai chiều,
  sg_rel 101.8k). nice, 2 luồng: 11 epoch × ~11 phút = **2 giờ** (đỉnh val ở epoch 6, 0.7356).
- **Kết quả test seed 0:** toàn pool R@1 / MRR **52.63 / 64.83**; gallery R@1 / MRR **62.99 / 74.04** (cao nhất đến nay);
  direct / indirect **79.23 / 72.84** (α 0.35).
- **So sánh (MRR, 1 seed):** toàn pool — adapter 63.26 · qa 64.30 · qa no_rel 64.98 · qa rel≥5 64.43 · **qa+CN 64.83**;
  indirect — adapter 72.06 · qa 72.12 · no_rel 72.34 · rel≥5 72.34 · **qa+CN 72.84** (+0.7 so với qa không CN).
- **Nhận xét:** ConceptNet là thứ duy nhất nhích được nhóm indirect (+0.5–0.8), nhưng chênh vẫn trong nhiễu 1 seed và
  toàn pool chưa vượt no_rel. Vì sg_rel không giúp, cấu hình đáng thử là ConceptNet KHÔNG sg_rel → thêm biến thể `no_sg`
  (`vg_qa_cn_no_sg`). Kết luận cần 3 seed → Kaggle (CPU 2 giờ/lượt).

## #8 · 2026-11-18 · `vg_qa --set extra=data/vg_bundle/graph_extra_cn` seed 0 — QA + ConceptNet

- **Cấu hình:** như #5 + 16 413 cạnh ConceptNet (32 826 cạnh hai chiều), 1000 đỉnh khái niệm (3000 nhãn/khái niệm), nice 2 luồng,
  ~2 giờ (11 epoch, val MRR đỉnh 0.7356 ở epoch 6).
- **Kết quả test seed 0:** toàn pool R@1 / MRR **52.63 / 64.83** (QA #5: 52.11 / 64.30; no_rel #6: 53.01 / 64.98; Adapter 63.26);
  gallery MRR 74.04 (#5: 73.71); direct / indirect **79.23 / 72.84** (#5: 80.54 / 72.12; Adapter 73.46 / 72.06). α = 0.35.
- **Nhận xét:** ConceptNet là thứ đầu tiên nhích được nhóm **indirect** (+0.7 so với #5, +0.8 so với Adapter) nhưng direct giảm
  1.3; tổng toàn pool +0.5 so với #5 và ngang no_rel. Một seed — chưa đủ để khẳng định; cần 3 seed cho Adapter / QA / QA-no_rel /
  QA+CN, và ablation QA+CN no_rel (ConceptNet không kèm sg_rel) vì sg_rel có thể đang làm nhiễu.
- **Output:** `runs/vg_qa/20261117-142744_lab_vg_qa-cn-seed0/` (UI mặc định có thể trỏ `MMXAI_RUN` vào đây).

## #7 · 2026-11-17 · `vg_qa --set graph.min_rel_count=5` seed 0 — sg_rel thưa cũng không giúp; ConceptNet đã có

- **sg_rel thưa** (≥5 lần trên train: 13.4k cạnh, ~7 láng giềng/nhãn thay vì 49): toàn pool R@1 / MRR 52.25 / 64.43,
  gallery MRR 73.63, direct / indirect 79.15 / 72.34 → nằm giữa full (64.30) và no_rel (64.98); không cải thiện.
- **ConceptNet:** S3 của ConceptNet ngắt kết nối liên tục (3 lần). Dùng bản tiếng Anh trên Hugging Face
  `peandrew/conceptnet_en_simple` (3.42M cạnh en–en, cột rel/arg1/arg2/weight, cùng URI `/c/en/...`); `fetch_conceptnet.py`
  đọc được `.parquet`. Lọc (IsA, PartOf, UsedFor, AtLocation, CapableOf, HasA, Synonym, weight ≥ 1, +1000 khái niệm):
  16 413 cạnh (IsA 6029, AtLocation 5581, Synonym 2239, UsedFor 1186, PartOf 935, HasA 231, CapableOf 212) →
  `data/vg_bundle/graph_extra_cn/` (tách khỏi `graph_extra/` để các lượt trước tái lập được).

## #6 · 2026-11-17 · `vg_qa_ablation_no_rel` seed 0 — cạnh nhãn–nhãn (sg_rel) không giúp

- **Cấu hình:** như #5, bỏ toàn bộ cạnh sg_rel (chỉ còn contains ảnh–nhãn; GNN trên nhãn chỉ còn self-loop). nice, 2 luồng, 18 phút.
- **Kết quả:** toàn pool R@1 / MRR **53.01 / 64.98** (full #5: 52.11 / 64.30); gallery MRR 73.81 (full 73.71);
  direct / indirect 80.13 / 72.34 (full 80.54 / 72.12).
- **Nhận xét:** bỏ cạnh nhãn–nhãn KHÔNG làm giảm, còn nhỉnh hơn (1 seed, chênh nằm trong nhiễu). Phần tăng của #5 đến từ
  cạnh contains ảnh–nhãn (đồ thị hai phía, 1 bước) + khớp khái niệm có học, không phải từ lan truyền 2 bước qua sg_rel.
  Giả thuyết: sg_rel gộp từ đồng xuất hiện trên train dày (~50 láng giềng/nhãn) và nhiễu → GNN làm nhòe danh tính nhãn.
- **Tiếp theo:** ConceptNet (quan hệ ngữ nghĩa IsA/PartOf… thưa hơn) cho nhóm indirect; thử sg_rel thưa (`min_rel_count`
  cao); 3 seed cho full / no_rel / no_edge để kết luận chắc.

## #5 · 2026-11-17 · `vg_qa` — điểm đồ thị phụ thuộc truy vấn (khái niệm của câu ↔ nhãn của ảnh), seed 0

- **Mục đích:** sửa nguyên nhân của #4 (gộp nhãn thành một vector mất thông tin khớp). `graph/qa.py`: GNN chỉ trên đồ thị nhãn,
  m(c,i) = max mềm cos(u_c, H_l) trên nhãn l của ảnh, s_G = trung bình theo khái niệm của caption (`concepts`, 100%
  nằm trong bộ 2000 nhãn); score = adapter + w·s_G. Test: no_edge ⇒ s_G = 0; đường đánh giá nhanh = đường huấn luyện.
- **Cấu hình:** GAT, chưa ConceptNet, batch 512. Máy bị lag khi 4 lượt song song → chạy tuần tự `nice`, 2 luồng (~44 phút/lượt).
- **Kết quả test seed 0 (toàn pool R@1 / MRR · gallery 1000 MRR):**
  CLIP 48.31 / 60.40 · 69.79 — khớp nhãn CLIP không học (untrained, không adapter) 48.40 / 60.47 · 69.93 —
  CLIP+Adapter (qa no_edge) 51.13 / 63.26 · 72.33 — **CLIP+Adapter+QA-Graph 52.11 / 64.30 · 73.71** (α 0.25).
  MRR theo nhóm (gallery): adapter direct 73.46 / indirect 72.06 → qa **direct 80.54** / indirect 72.12.
- **Nhận xét:** lần đầu đồ thị thêm được trên adapter: +1.04 MRR toàn pool, +1.4 gallery, +1.0 R@1 (1 seed). Toàn bộ phần
  tăng nằm ở nhóm **direct** (+7.1); indirect không đổi → chưa có bằng chứng lan truyền qua sg_rel giúp câu gián tiếp
  (giống AdaMF-MAT). Khớp nhãn không học gần như vô dụng (+0.07) → phần học (u_c, τ, w) là cần thiết.
- **Tiếp theo:** `no_rel` (bị ngắt khi phiên kết thúc) để biết cạnh nhãn–nhãn có góp gì; ConceptNet cho nhóm indirect;
  3 seed; kiểm định ghép cặp với adapter.
- **Output:** `runs/vg_qa/20261117-032408_lab_vg_qa-seed0/`, `runs/vg_qa_ablation_no_edge/20261117-030513_*`

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
  (+1.06). Giả thuyết: g_i gộp ~17 nhãn của ảnh thành MỘT vector không phụ thuộc truy vấn → mất thông tin
  "nhãn nào khớp khái niệm nào của câu", thứ mà khớp tên khai thác được; CLIP ảnh vốn đã mã hóa "túi đối tượng".
- **Tiếp theo:** điểm đồ thị phụ thuộc truy vấn (khái niệm của caption `concepts` ↔ nhãn của ảnh qua đường đi trong đồ thị),
  thêm ConceptNet; cùng bộ ablation.
- **Output:** `runs/vg_*/2026111[67]-*split*`

## #3 · 2026-11-16 · vg_ablation_no_edge / vg_ablation_shuffle (`--seed 0 --tag seed0`) — đồ thị CHƯA đóng góp

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
- **Output:** `runs/vg_ablation_no_edge/20261116-150308_lab_vg_ablation_no_edge-seed0/`,
  `runs/vg_ablation_shuffle/20261116-150308_lab_vg_ablation_shuffle-seed0/`

## #2 · 2026-11-15 · 20261115-082025_lab_vg_graph-seed0-dau-tien — dữ liệu THẬT, 1 seed (chưa phải số báo cáo)

- **Mục đích:** lượt CLIP+Graph đầu tiên trên bundle VG (VG∩COCO, Karpathy split, train 15k ảnh /
  75k caption, val 2126, test 2138), đánh giá bằng `mmxai_eval.Protocol` (gallery 1000 theo seed + full pool,
  `fuse` z-score, `tune_alpha` trên val bước 0.05).
- **Kiểm tra trước:** `vg_clip` khớp tuyệt đối baseline (R@1 57.36, MRR 69.23 mean 3 gallery; full 48.31 / 60.40);
  `vg_untrained` (GNN khởi tạo đơn vị) cho đúng số CLIP → đường dữ liệu đúng.
- **Cấu hình:** GAT 2 tầng, hidden 256, 4 head; đồ thị 19 264 ảnh + 2 000 nhãn (`object_vocab_train`), 334k cạnh contains,
  98.8k cạnh sg_rel (train, `min_rel_count=1`), CHƯA có ConceptNet. batch 512, lr 1e-3, tau 0.05, patience 4. CPU 4 luồng.
- **Kết quả test (seed 0):** gallery 1000: CLIP R@1 58.17 / MRR 69.79 → **60.45 / 72.15** (+2.3 / +2.4);
  full pool: 48.31 / 60.40 → **50.94 / 63.19** (+2.6 / +2.8). α = 0.35. Nhóm direct / indirect (định nghĩa):
  MRR 72.84 / 71.99.
- **Nhận xét:** cải thiện thật trên dữ liệu thật. MRR val (graph) đạt đỉnh ở epoch 1 rồi giảm → quá khớp sớm; còn chỗ dò
  lr/dropout/weight decay. CHƯA biết phần cải thiện đến từ cấu trúc đồ thị hay từ lớp chiếu học được → bắt buộc chạy
  `no_edge` và `shuffle` trên dữ liệu thật. 27 phút/seed trên CPU (≈4.4 phút/epoch).
- **Tiếp theo:** ablation no_edge + shuffle (seed 0), ConceptNet (A3), rồi 3 seed.
- **Output:** `runs/vg_graph/20261115-082025_lab_vg_graph-seed0-dau-tien/`

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
- **Tiếp theo:** nhận dữ liệu VG thật + features theo `docs/data-contract.md`; `make push --limit 5` trên Kaggle.
- **Output:** `runs/clip_graph/20261114-04*` (local, không commit).

## #0 · 2026-11-14 · khởi tạo

- Dựng harness bằng skill `paper-lab`; `make test` và `make local EXP=smoke` chạy được.
- Kế hoạch: `docs/plan.md`. Bảng cần tái lập: `docs/paper_numbers.yaml`.
