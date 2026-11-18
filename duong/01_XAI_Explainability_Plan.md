# Plan 1 — Explainability cho Knowledge Graph Text-to-Image Retrieval

## 1. Mục tiêu và phạm vi

- [ ] Xây dựng module giải thích **từng cặp truy vấn văn bản – ảnh** bằng các cạnh, đường đi hoặc đồ thị con **thực sự tham gia vào phép tính điểm truy xuất**.
- [ ] Đánh giá định lượng ba khía cạnh: **fidelity**, **sparsity**, **validity**.
- [ ] So sánh can thiệp lên cạnh quan trọng với can thiệp lên cạnh ngẫu nhiên.
- [ ] Phân tích tối thiểu **10 truy vấn**, gồm trường hợp truy xuất đúng và sai; chuẩn bị **5 case thành công + 5 case thất bại**.
- [ ] Tạo notebook minh họa: nhập truy vấn → kết quả top-k → explanation → điểm/thứ hạng thay đổi sau can thiệp.
- [ ] Cho phép chạy độc lập bằng scorer mẫu trước khi tích hợp GNN/reranker của nhóm; **không** sử dụng kết quả scorer mẫu làm kết luận thực nghiệm cuối.

**Ngoài phạm vi:** huấn luyện mô hình thị giác mới; xây thêm nhiều explainer; xây web app đầy đủ; đánh giá người dùng diện rộng.

## 2. Chuẩn hóa giao diện để không phụ thuộc các công đoạn khác

- [ ] Định nghĩa input thống nhất:
  - `query_id`, `query_text`, `query_type` (`direct`/`indirect`)
  - `candidate_image_ids` (một tập cố định cho phép so sánh thứ hạng)
  - `graph` (node ID, edge ID, source/target, relation type, provenance)
  - `clip_scores` (cố định khi can thiệp vào graph)
  - `model_checkpoint`, `alpha`, `seed`, `split_id`
- [ ] Định nghĩa interface `score(query, candidate_ids, graph) -> {image_id: {clip_score, graph_score, final_score}}`.
- [ ] Định nghĩa interface `explain(query, image_id, graph, scorer) -> [edge_id/path_id, contribution]`.
- [ ] Định nghĩa interface `evaluate(query, image_id, explanation, scorer) -> metrics`.
- [ ] Chuẩn hóa ID cạnh để không nhầm giữa cạnh quan hệ và cạnh ảnh–thực thể; lưu loại cạnh và nguồn gốc.
- [ ] Viết scorer mẫu minh bạch: ví dụ graph score là tổng trọng số các đường đi query–entity–image phù hợp, chuẩn hóa về cùng thang với CLIP score.
- [ ] Tạo adapter riêng để thay scorer mẫu bằng GCN/GAT/GraphSAGE hoặc phương pháp của nhóm, không sửa mã metric.

**Điều kiện bắt buộc:** khi xóa cạnh để kiểm tra fidelity, phải **tính lại graph score/forward pass với graph bị can thiệp**. Không chỉ xóa cạnh khỏi hình vẽ rồi giữ embedding/score cũ.

## 3. Xây dựng bộ truy vấn và protocol

- [ ] Sử dụng bộ ≥20 truy vấn đánh giá chung của nhóm nếu có; nếu chưa có, tạo bộ độc lập **10 truy vấn trực tiếp + 10 truy vấn gián tiếp**.
- [ ] Với mỗi truy vấn, xác định `query_type`, ảnh liên quan và căn cứ xác nhận relevance; không suy relevance từ chính model đang đánh giá.
- [ ] Giữ nguyên candidate pool, các ảnh, CLIP scores, model weights và `alpha` khi thực hiện các phép can thiệp.
- [ ] Phân biệt:
  - **Retrieval success/failure:** ảnh đúng có/không có trong top-k theo ground truth.
  - **Explanation faithful/unfaithful:** cạnh giải thích có/không có ảnh hưởng đáng kể tới prediction.
- [ ] Chỉ chọn siêu tham số (như `alpha`, số cạnh giải thích `k`, normalization) trên **validation**; không dùng bộ đánh giá cuối để tuning.
- [ ] Khi có thực nghiệm mô hình đã huấn luyện, chạy các thí nghiệm chính bằng **ít nhất 3 seed** và báo cáo trung bình ± độ lệch chuẩn; dùng cùng split với nhóm.

## 4. Trích xuất explanation

### 4.1. Phương pháp chính: edge/path perturbation

- [ ] Xác định đồ thị con cục bộ `G_local(q,i)` có khả năng ảnh hưởng đến điểm của ảnh `i` (ví dụ: neighborhood thực sự được GNN sử dụng, hoặc tập đường đi được scorer xét).
- [ ] Với từng cạnh ứng viên `e`, chạy lại scoring khi loại bỏ/mask cạnh đó:

  `contribution(e) = score(q,i,G) - score(q,i,G \ {e})`

- [ ] Ghi **riêng** đóng góp vào `graph_score` và `final_score`.
- [ ] Sắp xếp các cạnh có đóng góp dương theo mức ảnh hưởng; chọn `top-k` với `k ∈ {1, 3, 5}`.
- [ ] Gom các cạnh thành đường đi/đồ thị con hợp lệ. Nếu cạnh riêng lẻ không tạo thành đường đi liền mạch, báo rõ đó là **top edges**, không ngụy tạo path.
- [ ] Nếu cơ chế score sử dụng **đường đi** thay vì cạnh, đo đóng góp path bằng ablation path và theo dõi các cạnh chồng lấn.
- [ ] Lưu explanation cùng các giá trị `baseline_score`, `perturbed_score`, `contribution`, `provenance`.

**Lưu ý:** đóng góp cạnh đơn lẻ không cộng tuyến tính khi mô hình có tương tác phi tuyến. Không diễn giải tổng đóng góp cạnh là decomposition chính xác nếu chưa kiểm chứng.

### 4.2. Phương pháp tham chiếu (tùy chọn)

- [ ] Chỉ tích hợp GNNExplainer/PyG Explainer nếu scorer của nhóm tương thích; áp dụng **cùng** bộ metric và cùng candidate pool.

## 5. Cài đặt metric và phép thử

### 5.1. Fidelity — mức trung thực đối với mô hình

- [ ] Đo độ giảm điểm khi xóa tập cạnh explanation `E_exp`:

  `FidelityDeletion = S(q,i,G) - S(q,i,G \ E_exp)`

- [ ] Tính **graph-score drop** và **final-score drop** riêng; ưu tiên dùng score trước/sau trên cùng thang đo.
- [ ] Đo `RankDrop = rank_after(image_i) - rank_before(image_i)` trên **cùng candidate pool**; số dương nghĩa là tụt hạng.
- [ ] So sánh với **random-edge deletion**: lấy cùng số cạnh từ cùng `G_local`, lặp bằng nhiều random seed, báo trung bình và độ lệch chuẩn.
- [ ] Tùy chọn tính **sufficiency**: chỉ giữ `E_exp` (và các node/cạnh bắt buộc về mặt schema) để kiểm tra mức bảo toàn điểm so với full graph.
- [ ] Không khẳng định fidelity tốt chỉ vì score giảm: kiểm tra mức giảm có lớn hơn baseline xóa cạnh ngẫu nhiên hay không.

### 5.2. Sparsity — độ cô đọng của giải thích

- [ ] Dùng mẫu số là **cạnh thuộc đồ thị con cục bộ hợp lệ** `E_local`, không phải toàn bộ KG:

  `Sparsity = 1 - |E_exp| / |E_local|`

- [ ] Báo thêm `#explained_edges`, `#explained_paths`, `average_path_length`.
- [ ] So sánh theo `k = 1, 3, 5` để thấy trade-off giữa sparsity và fidelity.
- [ ] Không coi sparsity cao là tốt nếu fidelity/validity thấp.

### 5.3. Validity — tính hợp lệ và có căn cứ

- [ ] `StructuralValidity = valid_explanation_edges / total_explanation_edges`.
- [ ] Kiểm tra tự động mỗi cạnh:
  - Edge ID thực sự tồn tại trong graph đầu vào.
  - Head, relation, tail tuân theo schema đã định nghĩa.
  - Đường đi liền mạch và kết nối các node liên quan tới truy vấn/ảnh như đã mô tả.
  - Không dùng cạnh được tạo sau khi biết kết quả test để giải thích ngược.
- [ ] `EvidenceSupportRate = supported_explanation_edges / total_explanation_edges`.
- [ ] Kiểm tra mức được hỗ trợ từ caption, nguồn triple, metadata ảnh hoặc xác nhận thị giác; lưu `support_source` và trạng thái `supported`/`unsupported`/`uncertain`.
- [ ] Rà soát thủ công **mẫu nhỏ của 10 case studies** để kiểm chứng semantic validity; không phải annotation toàn tập.
- [ ] Báo **Structural Validity** và **Evidence Support** tách biệt, không gộp thành một con số không rõ nghĩa.

## 6. Ma trận thực nghiệm tối thiểu

| Thực nghiệm | Biến thay đổi | Chỉ số cần báo cáo |
|---|---|---|
| Full graph vs delete explained edges | Xóa cạnh explanation | Graph/final score drop, rank drop |
| Explained vs random edge deletion | Cùng số cạnh, khác cách chọn | Fidelity tương đối, độ lệch chuẩn |
| `k = 1, 3, 5` | Kích thước explanation | Fidelity–sparsity trade-off |
| Direct vs indirect queries | Loại truy vấn | Fidelity, validity, retrieval success |
| Success vs failure cases | Tính đúng/sai của retrieval | Phân tích lỗi và giải thích |

## 7. Đầu ra và báo cáo

- [ ] `xai/scorer_adapter.py`: scorer mẫu + adapter scorer thật.
- [ ] `xai/explain.py`: trích xuất top edges/paths dựa trên đóng góp.
- [ ] `xai/evaluate_xai.py`: fidelity, sparsity, validity; random baseline.
- [ ] `xai/queries.csv`: truy vấn, loại truy vấn, ground truth/relevance provenance.
- [ ] `experiments/xai_results.csv`: một dòng cho mỗi `(query_id, image_id, method, seed, k, perturbation)`.
- [ ] `experiments/case_studies/`: ≥10 case, gồm 5 success và 5 failure.
- [ ] `demo_xai.ipynb`: hiển thị top-k ảnh, graph/path, edge contributions, score/rank trước và sau perturbation.
- [ ] Biểu đồ: fidelity explained vs random; fidelity–sparsity theo k; metric theo direct/indirect.
- [ ] README: lệnh chạy, dependencies, schema, cách chọn split/seed và giới hạn của từng metric.

### Tiêu chí nghiệm thu

- [ ] Explanation được tạo từ **cùng graph và cơ chế score** đang dùng cho retrieval.
- [ ] Xóa/mask cạnh explanation làm scorer **thực sự chạy lại**.
- [ ] Có đủ ba khía cạnh fidelity, sparsity, validity trên **≥10 truy vấn**.
- [ ] Có baseline ngẫu nhiên, ít nhất 3 seed cho thực nghiệm chính, và bằng chứng 5 case đúng/5 case sai.
- [ ] Notebook và các tập kết quả có thể chạy lại, không phụ thuộc giao diện web.

## 8. Phối hợp tích hợp — chỉ yêu cầu các artifact thiết yếu

- **Từ bộ phận baseline:** `image_ids`, CLIP scores, candidate selection, split IDs và metric retrieval.
- **Từ bộ phận graph/GNN:** graph schema, graph builder, model checkpoint và hàm forward có thể nhận graph bị mask/xóa cạnh.
- **Từ bộ phận phương pháp mới:** scorer tốt nhất và bộ truy vấn trực tiếp/gián tiếp dùng chung.
- **Nếu chưa nhận được:** chạy đầy đủ unit test và phương pháp perturbation với scorer mẫu; để trống kết luận định lượng về GNN thật đến khi tích hợp được scorer.
