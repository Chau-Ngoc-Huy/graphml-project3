# XAI visual review packs

- `baseline_case_review_10.xlsx`: 10 case từ snapshot cũ, 20 ảnh GT/top-1 nhúng trong tab **Case review**. Tab **Step review** chứa cạnh được mô hình chọn ở seed 0, k=3. Cạnh `contains` chỉ là gán đối tượng, không phải bằng chứng cho hành động hoặc quan hệ.
- `queries_relational_candidates.csv`: 40 caption test đã chọn tất định trước path XAI (10 direct nguồn + 10 indirect nguồn mỗi bộ), kèm CLIP embedding row và concept của chính bundle. `queries_relational_candidates.sha256` khóa danh sách. Ảnh âm khó được gợi ý theo overlap đối tượng trong top-100; nó **chưa** là ảnh âm đã xác minh.
- `query_candidates_40.xlsx`: 40 cặp ảnh dương/âm đề xuất, URL nguồn có thể mở trực tiếp. Chạy notebook `01_xai_audit_and_query_review.ipynb` trên Colab để tải và nhúng ảnh vào Excel nếu các URL còn truy cập được.

Mỗi người chấm làm trên **một bản sao** workbook. Ở tab **Case review**, điền `verified_group` (`direct`/`relational`), ba thành phần claim, `gt_relevance`, `retrieved_relevance`, `hard_negative_same_objects`, `relation_differs`, `visual_reason`, `reviewer` và `review_status`. Dùng `supported`/`contradicted`/`uncertain` cho bốn trường phán đoán. Chỉ đặt `review_status=approved` sau khi ảnh dương hỗ trợ toàn truy vấn, ảnh âm bác bỏ truy vấn, có cùng đối tượng và khác quan hệ/vai trò. Đọc tab **Rubric** trước khi chấm.

Giải quyết bất đồng trên một bản workbook riêng. Đặt đường dẫn bản đó vào `REVIEWED_WORKBOOK` trong notebook 01 để xuất `queries_relational_verified.csv`; nguồn ứng viên không bị ghi đè. Notebook 02 chặn chế độ `verified40` nếu thiếu nhãn hoặc có dưới 10 direct + 10 relational đã xác minh cho bất kỳ dataset nào.

Notebook 03 tạo workbook mới `validity_case_review_10.xlsx` gồm 5 success và 5 failure từ path run. Hai người chấm độc lập từng bước và toàn path; sau đó nhập hai workbook để tính đồng thuận và tạo bản hòa giải. `validity_annotations.csv` ban đầu là template trống. Chỉ báo logical/visual support sau khi nạp bản hòa giải đầy đủ.
