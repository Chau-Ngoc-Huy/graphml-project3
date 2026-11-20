/**
 * Narrative content. Numbers here are transcribed from the project logs and carry their source;
 * everything that can be computed is computed from public/data instead.
 */
export const PROBLEM = {
  title: "Truy vấn đa phương thức có giải thích bằng đồ thị",
  oneLiner: "Cho một câu mô tả, tìm đúng bức ảnh trong kho, rồi xem tri thức relation (scene graph, ConceptNet) có giúp ranking tốt hơn CLIP không — và vì sao.",
  problem:
    "CLIP xếp ảnh theo độ tương đồng vectơ: nhanh và mạnh, nhưng không dùng cấu trúc relation giữa object và không cho biết vì sao một ảnh được trả về.",
  question: "Tri thức relation giúp query tốt hơn ở đâu, nhất là với query gián tiếp — và cải thiện có thật sự đến từ cấu trúc graph?",
  objectives: [
    { k: "Query", v: "Baseline CLIP đóng băng + giao thức đánh giá chung (cùng split, cùng gallery, cùng top-100)." },
    { k: "Graph", v: "Graph ảnh–label–concept từ scene graph Visual Genome + ConceptNet; GCN / GAT / GraphSAGE rerank." },
    { k: "Phương pháp mới", v: "AdaMF-MAT reproduction (LREC-COLING 2024) và đặt vào cùng pipeline." },
    { k: "Explanation", v: "Path / subgraph đóng góp vào điểm, đánh giá fidelity · sparsity · validity." },
  ],
};

export interface Stage {
  id: string;
  title: string;
  short: string;
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
    id: "data", title: "Dữ liệu & split cố định", short: "Dữ liệu",
    what: "Ghép Visual Genome (scene graph) với MS-COCO (5 caption người viết / ảnh) qua coco_id, chia train / val / test theo Karpathy.",
    why: "Mọi model phải chạy trên cùng một split để so sánh công bằng; thống kê graph chỉ được tính trên train để không rò rỉ.",
    inputs: ["scene_graphs.json (VG 1.4, 739 MB)", "image_data.json (coco_id, URL ảnh)", "dataset_coco.json (Karpathy)"],
    outputs: ["split.json", "images.jsonl (object + relation đã chuẩn hoá)", "captions.jsonl (nhóm direct / indirect)"],
    method: ["Karpathy train+restval → train (lấy mẫu 15 000, seed 0), val → val, test → test", "Chuẩn hoá tên object / vị từ", "Từ vựng 2 000 label đếm trên ảnh train"],
    status: "done", next: "clip",
  },
  {
    id: "clip", title: "Đặc trưng CLIP đóng băng + FAISS", short: "CLIP",
    what: "Mã hoá mọi ảnh và caption bằng OpenCLIP ViT-B/32 (laion2b) một lần, lưu lại; FAISS lấy top-100 candidate theo cosine.",
    why: "Tầng truy hồi nhanh, chung cho mọi phương pháp: các graph model chỉ rerank 100 candidate này.",
    inputs: ["Ảnh VG (stream qua URL)", "Caption COCO"],
    outputs: ["Đặc trưng ảnh 19 264 × 512", "Đặc trưng caption 96 381 × 512", "Top-100 candidate / query", "Baseline CLIP"],
    method: ["Không fine-tune", "Chuẩn hoá L2 → tích vô hướng = cosine", "3 fold gallery 1 000 ảnh theo seed + toàn pool"],
    status: "done", next: "concepts",
  },
  {
    id: "concepts", title: "Concept linking", short: "Concept",
    what: "Tách lemma của câu, khớp n-gram dài nhất với từ vựng label: 'Two cats on a desk' → {cat, desk}.",
    why: "Là cầu nối giữa văn bản và graph: graph score đo xem concept của câu có khớp label của ảnh (trực tiếp hay qua relation) hay không.",
    inputs: ["Caption", "Từ vựng 2 000 label (train)"],
    outputs: ["Concept của từng caption", "Nhóm query: direct / indirect / no_concept"],
    method: ["Lemma theo luật (không cần NLTK)", "Bỏ màu sắc / danh từ chung khi gán nhóm", "Nhóm chỉ để báo cáo, model không dùng"],
    status: "done", next: "graph",
  },
  {
    id: "graph", title: "Dựng graph ảnh – label – concept", short: "Graph",
    what: "Node: ảnh, label object, concept ConceptNet. Edge: ảnh–chứa–label, label–relation scene graph–label (train), label–ConceptNet–concept.",
    why: "Cho phép một câu nói 'child' tới được ảnh chỉ ghi 'boy' qua relation — giả thuyết cần kiểm chứng của đề.",
    inputs: ["images.jsonl", "ConceptNet (IsA, AtLocation, Synonym, UsedFor, PartOf, HasA, CapableOf)"],
    outputs: ["contains: 336 581 edge", "ConceptNet: 32 826 edge (2 chiều), 1 000 concept thêm", "3 000 node label + concept"],
    method: ["Relation scene graph chỉ gộp từ ảnh train", "ConceptNet lọc theo từ vựng thị giác", "AdaMF-MAT: MMKG 2 000 concept + 10 264 ảnh cho AdaMF-MAT"],
    status: "done", next: "gnn",
  },
  {
    id: "gnn", title: "GNN + graph score theo query", short: "GNN",
    what: "GNN (GCN / GAT / GraphSAGE) học biểu diễn label; graph score s_G = max mềm cos(concept của câu, label của ảnh), học chung với Adapter.",
    why: "Thiết kế đầu tiên gộp label của ảnh thành một vector và không thêm gì so với Adapter; điểm phụ thuộc query mới khai thác được 'label nào concept matching nào'.",
    inputs: ["Graph", "Đặc trưng CLIP của label ('a photo of a {label}')", "Caption train + hard negative top-100"],
    outputs: ["Graph score cho 100 candidate", "model.pt"],
    method: ["InfoNCE, early stopping theo MRR val", "3 seed (= gallery + khởi tạo)", "AdaMF-MAT: điểm = độ hợp lý của (ảnh, contains, concept)"],
    status: "done", next: "fusion",
  },
  {
    id: "fusion", title: "Score fusion, chọn α trên val", short: "Kết hợp",
    what: "score = α·z(cos) + (1−α)·z(s_G), z-score trong top-100; α quét 0→1 bước 0,05 chỉ trên tập val.",
    why: "Cosine CLIP dao động hẹp; chuẩn hoá z giúp α có nghĩa. Chọn trên val để không rò rỉ test.",
    inputs: ["Cosine CLIP", "Graph score"],
    outputs: ["Thứ rank cuối của 100 candidate", "α* cho từng cấu hình"],
    method: ["tune_alpha() trong mmxai_eval.py", "Gold image ngoài top-100 giữ rank CLIP"],
    status: "done", next: "eval",
  },
  {
    id: "eval", title: "Đánh giá & kiểm định", short: "Đánh giá",
    what: "R@1/5/10, MRR, nDCG@10 trên 3 fold gallery 1K và toàn pool 2 138 ảnh; tách direct / indirect; bootstrap ghép cặp theo query.",
    why: "Để biết cải thiện có ý nghĩa thống kê và đến từ đâu — ablation edge removal / shuffle là bằng chứng bắt buộc.",
    inputs: ["Thứ rank từng query của mọi model"],
    outputs: ["Bảng 3 seed mean ± std", "Ablation", "Khoảng tin cậy 95 %"],
    method: ["Cùng module đánh giá cho mọi người", "Paired bootstrap test trên cùng 10 696 query"],
    status: "done", next: "xai",
  },
  {
    id: "xai", title: "Explanation (xAI)", short: "Explanation",
    what: "Trích edge / path đóng góp vào điểm của từng (query, ảnh), xoá hoặc nhiễu chúng rồi chấm lại để đo fidelity.",
    why: "Đề yêu cầu explanation phải đến từ chính cơ chế ranking, không phải path 'nghe hợp lý' sinh sau.",
    inputs: ["Graph", "Scorer (model đã huấn luyện)", "≥ 10 query"],
    outputs: ["Path + đóng góp", "Fidelity / sparsity / validity"],
    method: ["Khung code trích path + nhiễu edge, có test", "explain.py: fidelity mới chạy trên dữ liệu giả"],
    status: "partial",
  },
];

/** Iterations of the graph model — from results/JOURNAL.md (MRR %, full test pool, seed 0 unless noted). */
export const GRAPH_JOURNEY = [
  { id: "#2", date: "15/11", title: "Lượt graph đầu tiên", graph: 63.19, ref: 60.4, refLabel: "CLIP",
    finding: "+2,8 MRR so với CLIP — có vẻ graph giúp.", verdict: "nghi ngờ" },
  { id: "#3", date: "16/11", title: "Ablation edge removal / shuffle", graph: 63.19, ref: 63.14, refLabel: "no_edge",
    finding: "Bỏ hết edge (63,14) hay edge shuffle (63,17) cho cùng mức tăng → toàn bộ phần tăng là của Adapter, không phải graph.", verdict: "bác bỏ" },
  { id: "#4", date: "17/11", title: "Tách Adapter khỏi graph branch", graph: 63.25, ref: 63.35, refLabel: "Adapter",
    finding: "Kiến trúc đúng trên dữ liệu giả, nhưng trên dữ liệu thật label graph không thêm gì: gộp label thành 1 vector làm mất 'label nào concept matching nào'.", verdict: "không giúp" },
  { id: "#5", date: "17/11", title: "Graph score phụ thuộc query (QA-Graph)", graph: 64.3, ref: 63.26, refLabel: "Adapter",
    finding: "Lần đầu graph thêm được trên Adapter: +1,04 MRR — toàn bộ ở nhóm direct.", verdict: "có tác dụng" },
  { id: "#6", date: "17/11", title: "Bỏ edge label–label (sg_rel)", graph: 64.98, ref: 64.3, refLabel: "QA-Graph",
    finding: "Không giảm mà còn nhỉnh hơn: phần tăng đến từ edge ảnh–chứa–label, không phải lan truyền qua relation.", verdict: "relation không giúp" },
  { id: "#8", date: "17/11", title: "+ ConceptNet", graph: 64.83, ref: 64.3, refLabel: "QA-Graph",
    finding: "Thứ duy nhất nhích được nhóm indirect (+0,7), nhưng trong nhiễu 1 seed.", verdict: "nhỏ" },
  { id: "#9", date: "18/11", title: "Chính thức: 11 cấu hình × 3 seed", graph: 64.89, ref: 63.14, refLabel: "Adapter (3 seed)",
    finding: "SAGE · contains + ConceptNet: +1,93 MRR (CI95 [1,49; 2,32]); edge shuffle tụt dưới Adapter (−0,41).", verdict: "kết luận" },
] as const;
