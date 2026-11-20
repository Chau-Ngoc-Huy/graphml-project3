/**
 * Narrative content. Numbers here are transcribed from the team's own logs and carry their source;
 * everything that can be computed is computed from public/data instead.
 */
export const PROBLEM = {
  title: "Truy vấn đa phương thức có giải thích bằng đồ thị",
  oneLiner: "Cho một câu mô tả, tìm đúng bức ảnh trong kho, rồi xem tri thức quan hệ (scene graph, ConceptNet) có giúp xếp hạng tốt hơn CLIP không — và vì sao.",
  problem:
    "CLIP xếp ảnh theo độ tương đồng vectơ: nhanh và mạnh, nhưng không dùng cấu trúc quan hệ giữa đối tượng và không cho biết vì sao một ảnh được trả về.",
  question: "Tri thức quan hệ giúp truy vấn tốt hơn ở đâu, nhất là với truy vấn gián tiếp — và cải thiện có thật sự đến từ cấu trúc đồ thị?",
  objectives: [
    { k: "Truy vấn", v: "Baseline CLIP đóng băng + giao thức đánh giá chung (cùng split, cùng gallery, cùng top-100)." },
    { k: "Đồ thị", v: "Đồ thị ảnh–nhãn–khái niệm từ scene graph Visual Genome + ConceptNet; GCN / GAT / GraphSAGE xếp hạng lại." },
    { k: "Phương pháp mới", v: "Tái lập AdaMF-MAT (LREC-COLING 2024) và đặt vào cùng pipeline." },
    { k: "Giải thích", v: "Đường đi / đồ thị con đóng góp vào điểm, đánh giá fidelity · sparsity · validity." },
  ],
};

export interface Stage {
  id: string;
  title: string;
  short: string;
  owner: string;
  what: string;
  why: string;
  inputs: string[];
  outputs: string[];
  method: string[];
  status: "done" | "partial" | "planned";
  next?: string;
}

export const STAGES: Stage[] = [
  {
    id: "data", title: "Dữ liệu & split cố định", short: "Dữ liệu", owner: "Thắng · Yến",
    what: "Ghép Visual Genome (scene graph) với MS-COCO (5 caption người viết / ảnh) qua coco_id, chia train / val / test theo Karpathy.",
    why: "Mọi mô hình phải chạy trên cùng một split để so sánh công bằng; thống kê đồ thị chỉ được tính trên train để không rò rỉ.",
    inputs: ["scene_graphs.json (VG 1.4, 739 MB)", "image_data.json (coco_id, URL ảnh)", "dataset_coco.json (Karpathy)"],
    outputs: ["split.json", "images.jsonl (đối tượng + quan hệ đã chuẩn hoá)", "captions.jsonl (nhóm direct / indirect)"],
    method: ["Karpathy train+restval → train (lấy mẫu 15 000, seed 0), val → val, test → test", "Chuẩn hoá tên đối tượng / vị từ", "Từ vựng 2 000 nhãn đếm trên ảnh train"],
    status: "done", next: "clip",
  },
  {
    id: "clip", title: "Đặc trưng CLIP đóng băng + FAISS", short: "CLIP", owner: "Thắng",
    what: "Mã hoá mọi ảnh và caption bằng OpenCLIP ViT-B/32 (laion2b) một lần, lưu lại; FAISS lấy top-100 ứng viên theo cosine.",
    why: "Tầng truy hồi nhanh, chung cho mọi phương pháp: các mô hình đồ thị chỉ xếp hạng lại 100 ứng viên này.",
    inputs: ["Ảnh VG (stream qua URL)", "Caption COCO"],
    outputs: ["Đặc trưng ảnh 19 264 × 512", "Đặc trưng caption 96 381 × 512", "Top-100 ứng viên / truy vấn", "Baseline CLIP"],
    method: ["Không fine-tune", "Chuẩn hoá L2 → tích vô hướng = cosine", "3 fold gallery 1 000 ảnh theo seed + toàn pool"],
    status: "done", next: "concepts",
  },
  {
    id: "concepts", title: "Liên kết câu → khái niệm", short: "Khái niệm", owner: "Thắng",
    what: "Tách lemma của câu, khớp n-gram dài nhất với từ vựng nhãn: 'Two cats on a desk' → {cat, desk}.",
    why: "Là cầu nối giữa văn bản và đồ thị: điểm đồ thị đo xem khái niệm của câu có khớp nhãn của ảnh (trực tiếp hay qua quan hệ) hay không.",
    inputs: ["Caption", "Từ vựng 2 000 nhãn (train)"],
    outputs: ["Khái niệm của từng caption", "Nhóm truy vấn: direct / indirect / no_concept"],
    method: ["Lemma theo luật (không cần NLTK)", "Bỏ màu sắc / danh từ chung khi gán nhóm", "Nhóm chỉ để báo cáo, mô hình không dùng"],
    status: "done", next: "graph",
  },
  {
    id: "graph", title: "Dựng đồ thị ảnh – nhãn – khái niệm", short: "Đồ thị", owner: "Huy · Khánh",
    what: "Đỉnh: ảnh, nhãn đối tượng, khái niệm ConceptNet. Cạnh: ảnh–chứa–nhãn, nhãn–quan hệ scene graph–nhãn (train), nhãn–ConceptNet–khái niệm.",
    why: "Cho phép một câu nói 'child' tới được ảnh chỉ ghi 'boy' qua quan hệ — giả thuyết cần kiểm chứng của đề.",
    inputs: ["images.jsonl", "ConceptNet (IsA, AtLocation, Synonym, UsedFor, PartOf, HasA, CapableOf)"],
    outputs: ["contains: 336 581 cạnh", "ConceptNet: 32 826 cạnh (2 chiều), 1 000 khái niệm thêm", "3 000 đỉnh nhãn + khái niệm"],
    method: ["Quan hệ scene graph chỉ gộp từ ảnh train", "ConceptNet lọc theo từ vựng thị giác", "Khánh: MMKG 2 000 khái niệm + 10 264 ảnh cho AdaMF-MAT"],
    status: "done", next: "gnn",
  },
  {
    id: "gnn", title: "GNN + điểm đồ thị theo truy vấn", short: "GNN", owner: "Huy",
    what: "GNN (GCN / GAT / GraphSAGE) học biểu diễn nhãn; điểm đồ thị s_G = max mềm cos(khái niệm của câu, nhãn của ảnh), học chung với Adapter.",
    why: "Thiết kế đầu tiên gộp nhãn của ảnh thành một vector và không thêm gì so với Adapter; điểm phụ thuộc truy vấn mới khai thác được 'nhãn nào khớp khái niệm nào'.",
    inputs: ["Đồ thị", "Đặc trưng CLIP của nhãn ('a photo of a {nhãn}')", "Caption train + hard negative top-100"],
    outputs: ["Điểm đồ thị cho 100 ứng viên", "model.pt"],
    method: ["InfoNCE, early stopping theo MRR val", "3 seed (= gallery + khởi tạo)", "AdaMF-MAT (Khánh): điểm = độ hợp lý của (ảnh, contains, khái niệm)"],
    status: "done", next: "fusion",
  },
  {
    id: "fusion", title: "Kết hợp điểm, chọn α trên val", short: "Kết hợp", owner: "Thắng · Huy",
    what: "score = α·z(cos) + (1−α)·z(s_G), z-score trong top-100; α quét 0→1 bước 0,05 chỉ trên tập val.",
    why: "Cosine CLIP dao động hẹp; chuẩn hoá z giúp α có nghĩa. Chọn trên val để không rò rỉ test.",
    inputs: ["Cosine CLIP", "Điểm đồ thị"],
    outputs: ["Thứ hạng cuối của 100 ứng viên", "α* cho từng cấu hình"],
    method: ["tune_alpha() trong mmxai_eval.py", "Ảnh đúng ngoài top-100 giữ hạng CLIP"],
    status: "done", next: "eval",
  },
  {
    id: "eval", title: "Đánh giá & kiểm định", short: "Đánh giá", owner: "Thắng · Huy · Khánh",
    what: "R@1/5/10, MRR, nDCG@10 trên 3 fold gallery 1K và toàn pool 2 138 ảnh; tách direct / indirect; bootstrap ghép cặp theo truy vấn.",
    why: "Để biết cải thiện có ý nghĩa thống kê và đến từ đâu — ablation bỏ / xáo cạnh là bằng chứng bắt buộc.",
    inputs: ["Thứ hạng từng truy vấn của mọi mô hình"],
    outputs: ["Bảng 3 seed mean ± std", "Ablation", "Khoảng tin cậy 95 %"],
    method: ["Cùng module đánh giá cho mọi người", "Kiểm định ghép cặp trên cùng 10 696 truy vấn"],
    status: "done", next: "xai",
  },
  {
    id: "xai", title: "Giải thích (xAI)", short: "Giải thích", owner: "Dương",
    what: "Trích cạnh / đường đi đóng góp vào điểm của từng (truy vấn, ảnh), xoá hoặc nhiễu chúng rồi chấm lại để đo fidelity.",
    why: "Đề yêu cầu giải thích phải đến từ chính cơ chế xếp hạng, không phải đường đi 'nghe hợp lý' sinh sau.",
    inputs: ["Đồ thị", "Scorer (mô hình đã huấn luyện)", "≥ 10 truy vấn"],
    outputs: ["Đường đi + đóng góp", "Fidelity / sparsity / validity"],
    method: ["Khung code + test (Dương)", "Huy: explain.py có fidelity trên dữ liệu giả"],
    status: "partial",
  },
];

/** Iterations of the graph model — from huy/results/JOURNAL.md (MRR %, full test pool, seed 0 unless noted). */
export const GRAPH_JOURNEY = [
  { id: "#2", date: "15/11", title: "Lượt đồ thị đầu tiên", graph: 63.19, ref: 60.4, refLabel: "CLIP",
    finding: "+2,8 MRR so với CLIP — có vẻ đồ thị giúp.", verdict: "nghi ngờ" },
  { id: "#3", date: "16/11", title: "Ablation bỏ / xáo cạnh", graph: 63.19, ref: 63.14, refLabel: "no_edge",
    finding: "Bỏ hết cạnh (63,14) hay xáo cạnh (63,17) cho cùng mức tăng → toàn bộ phần tăng là của Adapter, không phải đồ thị.", verdict: "bác bỏ" },
  { id: "#4", date: "17/11", title: "Tách Adapter khỏi nhánh đồ thị", graph: 63.25, ref: 63.35, refLabel: "Adapter",
    finding: "Kiến trúc đúng trên dữ liệu giả, nhưng trên dữ liệu thật đồ thị nhãn không thêm gì: gộp nhãn thành 1 vector làm mất 'nhãn nào khớp khái niệm nào'.", verdict: "không giúp" },
  { id: "#5", date: "17/11", title: "Điểm đồ thị phụ thuộc truy vấn (QA-Graph)", graph: 64.3, ref: 63.26, refLabel: "Adapter",
    finding: "Lần đầu đồ thị thêm được trên Adapter: +1,04 MRR — toàn bộ ở nhóm direct.", verdict: "có tác dụng" },
  { id: "#6", date: "17/11", title: "Bỏ cạnh nhãn–nhãn (sg_rel)", graph: 64.98, ref: 64.3, refLabel: "QA-Graph",
    finding: "Không giảm mà còn nhỉnh hơn: phần tăng đến từ cạnh ảnh–chứa–nhãn, không phải lan truyền qua quan hệ.", verdict: "quan hệ không giúp" },
  { id: "#8", date: "17/11", title: "+ ConceptNet", graph: 64.83, ref: 64.3, refLabel: "QA-Graph",
    finding: "Thứ duy nhất nhích được nhóm indirect (+0,7), nhưng trong nhiễu 1 seed.", verdict: "nhỏ" },
  { id: "#9", date: "18/11", title: "Chính thức: 11 cấu hình × 3 seed", graph: 64.89, ref: 63.14, refLabel: "Adapter (3 seed)",
    finding: "SAGE · contains + ConceptNet: +1,93 MRR (CI95 [1,49; 2,32]); xáo cạnh tụt dưới Adapter (−0,41).", verdict: "kết luận" },
] as const;

export const TEAM = [
  { name: "Thắng", role: "Nhiệm vụ 1 · Truy vấn & đánh giá chuẩn", done: ["OpenCLIP + FAISS", "Metric & module đánh giá dùng chung", "Baseline 3 fold, image → text", "Xuất dữ liệu theo hợp đồng nhóm"], status: "done" },
  { name: "Huy", role: "Nhiệm vụ 2 · Đồ thị, GNN, xếp hạng lại", done: ["Đồ thị VG + ConceptNet", "GCN / GAT / SAGE", "11 cấu hình × 3 seed (Kaggle)", "Ablation bỏ / xáo cạnh"], status: "done" },
  { name: "Khánh", role: "Nhiệm vụ 2 · Phương pháp mới + bộ truy vấn", done: ["Tái lập AdaMF-MAT (lệch < 1 điểm)", "AdaMF-MAT trên giao thức chung", "Bộ 24 truy vấn direct / indirect", "Kiểm định ghép cặp với Huy"], status: "done" },
  { name: "Dương", role: "Nhiệm vụ 3 · Giải thích (xAI)", done: ["Khung trích đường đi + nhiễu cạnh", "Metric fidelity / sparsity / validity (có test)"], status: "partial", gap: "Chưa chạy trên mô hình thật" },
  { name: "Yến", role: "Dữ liệu, split, báo cáo", done: [], status: "planned", gap: "Chưa có commit trong repo" },
] as const;
