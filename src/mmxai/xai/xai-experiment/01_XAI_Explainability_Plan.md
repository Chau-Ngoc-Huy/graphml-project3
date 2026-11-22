# Kế hoạch XAI cho direct text-to-image retrieval

## 1. Quyết định phạm vi

Tập 40 query được khóa trước khi xem path XAI. Người dùng đã duyệt toàn bộ nhãn trong [`query_candidates_40.xlsx`](review/query_candidates_40.xlsx): 39 query `direct`, một query `relational`; 34 query direct có GT `supported` đi vào thí nghiệm (18 VG, 16 PSG). Năm query direct có GT không `supported` và một query relational vẫn được giữ trong audit. Vì caption gần với nội dung ảnh, lượt mới tập trung vào **direct retrieval**: cạnh đối tượng nào trong graph thực sự ảnh hưởng đến điểm và hạng của ảnh top-1? Không đặt hạn mức query relational, không đo relational yield, không tuyên bố checkpoint hiểu predicate hoặc suy luận quan hệ.

[Lượt VG/PSG đã chạy](runs/qa_vg_psg_xai/xai-20261121-130842/analysis/report.md) là baseline cố định: 24 caption (12/bộ), 3 model seed, k=1/3/5, 216 dòng. Ở k=3, **102/108 cạnh VG** và **104/108 cạnh PSG** là `contains`; structural/path validity bằng 1 nhưng evidence support/known bằng 0. Vì vậy cần đánh giá căn cứ ảnh độc lập trước khi gọi explanation là có ý nghĩa ngữ nghĩa. Không viết lại baseline cũ.

## 2. Câu hỏi và giới hạn kết luận

1. **Fidelity đối với scorer:** xóa cạnh `contains` nối khái niệm query với ảnh đích, chạy lại QA scorer và phép fuse trên **cùng top-100**, đo đổi graph/QA/final score và hạng ảnh đích/GT. So với cạnh `contains` khác của **chính ảnh đích** có cùng số cạnh bị xóa. Cạnh quan hệ trong checkpoint vẫn giữ nguyên.
2. **Căn cứ đối tượng:** người chấm kiểm tra đối tượng tương ứng với cạnh có thực sự hiện trong ảnh và có liên quan đến query không. `contains` chỉ là bằng chứng đối tượng; nó không xác minh hành động, vai trò, vị trí hoặc toàn bộ caption.

Lưu vai trò `target_image_id` rõ ràng: top-1 trùng ID GT, top-1 khác ID GT, hay ảnh khác. **Khác ID GT không đồng nghĩa ảnh không phù hợp query**: một caption có thể mô tả nhiều ảnh. Với case ID mismatch, giảm điểm top-1 có thể cải thiện hạng GT, nhưng không tự chứng minh retrieval ban đầu sai về ngữ nghĩa. Checkpoint hiện tại dùng `edge_index` cho `label_h`, không mã hóa tên predicate làm đặc trưng cạnh; mọi kết luận về predicate nằm ngoài phạm vi này.

Người dùng xác nhận bằng mắt rằng cả năm ảnh top-1 trong nhóm GT-ID mismatch của workbook 10 case hỗ trợ query. Notebook ghi xác nhận này cho các ô relevance còn trống của đúng lượt `path-20261121-184857`, kèm `relevance_source=user_visual_confirmation`; nhãn đã điền trực tiếp trong workbook được ưu tiên. Mẫu chọn cân bằng theo ID không đại diện cho toàn bộ 34 query.

## 3. Tập query và cổng đánh giá

- Giữ 24 caption baseline để kiểm tra protocol mới trên dữ liệu cũ. Bộ 40 ứng viên mới vẫn giữ `source_group`, `verified_group`, query/GT/negative ID, CLIP embedding row, concepts, split và provenance. `source_group=indirect` có thể cùng tồn tại với `verified_group=direct`; nhãn nguồn chỉ là metadata.
- Đối với lượt mới, chọn **ít nhất 10 query direct đã duyệt cho mỗi dataset**. Điều kiện từng dòng: `verified_group=direct`, `gt_relevance=supported`, `review_status=approved`, claim và `visual_reason` đầy đủ, `reviewer` ghi nhận sự duyệt của người dùng. Bản nháp VLM chỉ là nguồn gợi ý; quyết định duyệt toàn workbook được lưu trong file chuẩn.
- Ảnh âm gợi ý là thông tin phụ. Không yêu cầu nó có cùng đối tượng hoặc khác quan hệ; không gọi nó là hard negative đã kiểm chứng và không dùng nó làm nhãn chính. Positive/GT phải được kiểm tra bằng mắt. Nếu muốn phân tích ảnh âm, báo riêng tình trạng relevance đã xác minh.
- Khóa tập đủ điều kiện và báo số dòng bị loại theo từng lý do **trước khi chạy perturbation**. Không chọn query theo score drop hoặc hạng sau can thiệp. Lấy tối đa 5 top-1 trùng ID GT và 5 top-1 khác ID GT để làm case study; nếu không đủ, báo thiếu thay vì thay query hậu kiểm. Đây là phân tầng theo **ID**, không phải nhãn semantic success/failure.
- Dùng caption ID, CLIP embedding/concepts, candidate pool, checkpoint và α của chính bundle. Không thay chuỗi caption mà giữ embedding cũ; không dùng tập test để tuning.

## 4. Trích xuất và can thiệp cạnh đối tượng

- Với mỗi `(query, top-1 image)`, xác định mọi cạnh `contains` của ảnh đích trước khi xếp hạng; lưu `E_eligible_contains`. Các cạnh nhãn–nhãn trong receptive field vẫn được audit nhưng không phải explanation chính.
- Ứng viên explanation là đường **một cạnh** từ nhãn/khái niệm query tới ảnh đích qua `contains`. Screen theo topology và giới hạn cố định, không theo kết quả perturbation hay nhãn ảnh. Ghi số cạnh đủ điều kiện, được screen và không có ứng viên.
- Tính đóng góp bằng xóa cạnh thật rồi chạy lại scorer. Với ngân sách 1/2/3 cạnh duy nhất, chọn cạnh có đóng góp biên dương; lưu cả trường hợp `no_object_path` và `no_positive_object_path`. Nếu nhiều cạnh chồng lấn, đếm cạnh một lần.
- Random control lấy các cạnh `contains` **khác trên cùng ảnh đích** và cùng ngân sách. Nếu không có tập độc lập, ghi N/A. Việc chọn explanation bằng score drop có thể làm advantage so với random cao hơn theo cơ chế chọn; báo đây là chỉ báo fidelity có điều kiện, không coi chênh lệch đó là kiểm định độc lập về tính đúng ngữ nghĩa.

## 5. Metric và kiểm chứng

| Trục | Cách đo và giới hạn |
|---|---|
| Fidelity | Graph/QA/final score drop và đổi hạng top-1/GT trước–sau; random cùng ảnh đích, loại cạnh và ngân sách. Báo `n_matched_queries`; không điền 0 khi thiếu random. |
| Coverage | Tỷ lệ **mọi case đã đăng ký** có cạnh query-matched và có ít nhất một cạnh đóng góp dương; `no_object_path` ở trong mẫu số. |
| Sparsity | `1 − số cạnh explanation duy nhất / E_eligible_contains`; báo cả tử số, mẫu số và screen coverage. Không tính sparsity khi không có explanation hoặc mẫu số bằng 0. |
| Structural validity | Edge ID, node type, chiều cạnh và đích ảnh hợp lệ; không suy ra căn cứ thị giác từ cấu trúc. |
| Object/visual support | Nếu chấm Step review, một workbook ghi `supported`/`contradicted`/`uncertain`, người chấm và lý do. Không tính đồng thuận hay yêu cầu hòa giải. Mức hỗ trợ quan hệ/toàn caption không được suy từ cạnh `contains`. |
| Top-1 semantic relevance | Với case top-1 khác ID GT, chấm riêng ảnh top-1 theo **toàn query** (`supported`/`contradicted`/`uncertain`) và ghi lý do. Exact-ID rank vẫn là metric benchmark; tỷ lệ semantic trên 10 case chọn cân bằng theo ID không đại diện cho toàn bộ 34 query. |

Bootstrap CI 95% theo **query**, giữ các seed trong cùng cụm; không coi các seed là các query độc lập. Báo kết quả VG và PSG riêng vì graph/GNN và thang điểm khác nhau. Chọn tối đa 10 case có ảnh (5 GT-ID match, 5 GT-ID mismatch nếu đủ); báo semantic support theo nhãn workbook hoặc xác nhận trực quan của người dùng. Object support chỉ báo nếu Step review có đủ nhãn.

## 6. Notebook và đầu ra

1. [`04_direct_query_audit.ipynb`](notebooks/04_direct_query_audit.ipynb): audit baseline, đọc workbook chuẩn, xuất `direct_query_eligibility_audit.csv` với lý do từng dòng và `queries_direct_verified.csv`. Không tái tạo workbook review từ đầu.
2. [`05_direct_object_xai_colab.ipynb`](notebooks/05_direct_object_xai_colab.ipynb): mặc định chạy 34 query direct đã duyệt; `baseline24` là lựa chọn kiểm tra code. Xuất `protocol.json`, `paths.jsonl`, `xai_results.csv`, `case_manifest.csv` trong lượt `path-*` mới.
3. [`06_direct_xai_analysis.ipynb`](notebooks/06_direct_xai_analysis.ipynb): báo coverage/fidelity/sparsity, CI theo query và exact-ID rate; tạo workbook 10 case có ảnh để chấm riêng relevance của top-1 và căn cứ đối tượng.

**Nghiệm thu lượt direct:** (1) mỗi dataset có ≥10 query direct/GT được người xác minh; (2) đủ provenance và giữ bản ghi bị loại; (3) mọi can thiệp dùng scorer thật trên candidate pool cố định; (4) no-path và random N/A ở đúng mẫu số; (5) sparsity dùng `E_eligible_contains`; (6) tối đa 10 case có ảnh, phân tầng theo GT-ID match/mismatch và ghi nguồn nhãn; (7) báo exact-ID và semantic relevance riêng; (8) cạnh đối tượng không chứng minh quan hệ hoặc toàn caption. Nghiên cứu relational/predicate-aware là công việc riêng sau lượt này.
