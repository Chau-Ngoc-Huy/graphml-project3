# Thuật ngữ dùng trong báo cáo

Mục đích: viết tiếng Việt tự nhiên, không dịch từng chữ từ tiếng Anh. Áp dụng cho mọi mục của `report/report.tex`.

## Nguyên tắc

1. **Lần đầu xuất hiện**: viết tiếng Việt, kèm thuật ngữ tiếng Anh in nghiêng trong ngoặc và một câu giải thích ngắn.
   Ví dụ: *ảnh đúng là **mẫu dương** (\emph{positive sample}), các ảnh khác trong batch là **mẫu âm** (\emph{negative sample}), tức các ví dụ sai để mô hình đối chiếu.*
2. **Các lần sau**: chỉ dùng tiếng Việt, không chú thích lại.
3. Thuật ngữ chuyên ngành đã quen thì **giữ nguyên tiếng Anh** (batch, epoch, adapter, GNN, GAT, GCN, GraphSAGE, ablation, baseline, seed, gallery, MRR, Recall@k, InfoNCE, MLP); không dịch ra thứ tiếng Việt mà người đọc vẫn phải đoán ngược về tiếng Anh.
4. Không viết chữ nào mà chính mình không giải thích được khi vấn đáp.

## Bảng thay thế

| Không dùng (dịch sát chữ) | Gốc tiếng Anh | Dùng | Ghi chú |
|---|---|---|---|
| CLIP đóng băng | frozen CLIP | CLIP **giữ nguyên trọng số** (không huấn luyện thêm) | lần đầu kèm *frozen* |
| điểm cuối của ứng viên $i$ | final score of candidate $i$ | **điểm xếp hạng** của ảnh $i$ / "ảnh $i$ được xếp hạng theo điểm…" | |
| ứng viên | candidate | **ảnh ứng viên** (lần đầu), sau đó "ứng viên" | top-$K$ ảnh CLIP chọn để xếp hạng lại |
| toàn pool test | full test pool | **toàn bộ ảnh test** | lần đầu kèm *full pool* |
| fold gallery, 3 fold | gallery fold | **gallery** thứ $s$; "3 lần lấy mẫu gallery" / "3 seed" | gallery = tập ảnh tìm kiếm 1 000 ảnh lấy ngẫu nhiên |
| (gallery) | gallery | **tập ảnh tìm kiếm (gallery)** lần đầu, sau đó "gallery" | |
| nền adapter, nền so sánh, trên nền adapter | adapter baseline | **mốc CLIP + adapter**, **mốc so sánh**, "khi đã có adapter" | |
| lớp chiếu | projection layer | **phép chiếu (tuyến tính) học được (adapter)** | |
| cổng $\gamma$ | gate | **hệ số trộn** $\gamma$ | |
| cực đại mềm | soft maximum | **max mềm (log-sum-exp)** | |
| lô | batch | **batch** | |
| đầu chú ý | attention head | **head chú ý (attention head)** | |
| kiên nhẫn 4 | patience 4 | dừng sớm khi **4 epoch liên tiếp** không cải thiện | |
| self-loop | self-loop | **cạnh tự nối (self-loop)** | |
| ảnh dương | positive / relevant image | **ảnh liên quan** (bộ 24 truy vấn) | |
| mẫu âm, mẫu dương | negative / positive sample | giữ **mẫu âm / mẫu dương**, lần đầu kèm tiếng Anh + giải thích | AdaMF-MAT: "bộ ba sai (mẫu âm)" |
| mẫu âm đối kháng | adversarial negative sampling | **bộ ba sai sinh có chủ đích cho sát bộ ba đúng** (*modality-adversarial training*) | |
| độ / tính hợp lý của bộ ba | triple plausibility | **điểm mức đúng của bộ ba** (*plausibility*) | |
| tất định | deterministic | **không phụ thuộc seed** | |
| đối sánh tương phản | contrastive learning | **học tương phản (contrastive learning)** | |
| túi đối tượng | bag of objects | **tập đối tượng có trong ảnh (bag of objects)** | |
| presence-only | presence-only baseline | **baseline chỉ dựa vào sự có mặt của đối tượng** | |
| làm nhòe danh tính nhãn | blur label identity | **làm các nhãn khó phân biệt với nhau** | |
| giao thức (đánh giá) | evaluation protocol | **quy trình đánh giá**; "gán tự động"; "cùng dữ liệu và cách chia" | |

## Tên gọi cố định

- Bộ dữ liệu: **VG∩COCO**, **PSG∩COCO**.
- Mô hình: **CLIP**, **CLIP + Adapter**, **QA-Graph**, **CLIP + khớp tên**, **AdaMF-MAT**.
- Nhóm truy vấn: **truy vấn trực tiếp / gián tiếp** (direct / indirect).
- Ablation: **phân tích loại bỏ (ablation)** lần đầu; biến thể giữ tên mã: `no_edge`, `no_rel`, `no_sg`, `shuffle` (xáo cạnh).
- Số liệu: MRR tính theo %, **dấu phẩy thập phân** (64,76).
