# XAI visual review packs

- `baseline_case_review_10.xlsx`: 10 case từ snapshot cũ, 20 ảnh GT/top-1 nhúng trong tab **Case review**. Tab **Step review** chứa cạnh được mô hình chọn ở seed 0, k=3. Cạnh `contains` chỉ là gán đối tượng, không phải bằng chứng cho hành động hoặc quan hệ.
- `queries_relational_candidates.csv`: 40 caption test đã chọn tất định trước path XAI (10 direct nguồn + 10 indirect nguồn mỗi bộ), kèm CLIP embedding row và concept của chính bundle. `queries_relational_candidates.sha256` khóa danh sách. Ảnh âm khó được gợi ý theo overlap đối tượng trong top-100; nó **chưa** là ảnh âm đã xác minh.
- `query_candidates_40.xlsx`: 40 cặp ảnh dương/âm đề xuất, URL nguồn có thể mở trực tiếp. Chạy notebook `01_xai_audit_and_query_review.ipynb` trên Colab để tải và nhúng ảnh vào Excel. Notebook thử URL Google Cloud Storage chính thức cho COCO trước, kiểm tra JPEG và ghi URL thực tế, SHA256, trạng thái TLS vào `image_download_manifest.csv`. Nếu một ảnh không tải được, liên kết nguồn vẫn còn trong workbook.

Mỗi người chấm làm trên **một bản sao** workbook. Ở tab **Case review**, điền `verified_group` (`direct`/`relational`), ba thành phần claim, `gt_relevance`, `retrieved_relevance`, `hard_negative_same_objects`, `relation_differs`, `visual_reason`, `reviewer` và `review_status`. Dùng `supported`/`contradicted`/`uncertain` cho bốn trường phán đoán. Chỉ đặt `review_status=approved` sau khi ảnh dương hỗ trợ toàn truy vấn, ảnh âm bác bỏ truy vấn, có cùng đối tượng và khác quan hệ/vai trò. Đọc tab **Rubric** trước khi chấm.

Giải quyết bất đồng trên một bản workbook riêng. Đặt đường dẫn bản đó vào `REVIEWED_WORKBOOK` trong notebook 01 để xuất `queries_relational_verified.csv`; nguồn ứng viên không bị ghi đè. Notebook 02 chặn chế độ `verified40` nếu thiếu nhãn hoặc có dưới 10 direct + 10 relational đã xác minh cho bất kỳ dataset nào.

Notebook 03 tạo workbook mới `validity_case_review_10.xlsx` gồm 5 success và 5 failure từ path run. Hai người chấm độc lập từng bước và toàn path; sau đó nhập hai workbook để tính đồng thuận và tạo bản hòa giải. `validity_annotations.csv` ban đầu là template trống. Chỉ báo logical/visual support sau khi nạp bản hòa giải đầy đủ.

## Gợi ý nhãn bằng Gemini trên máy local

Mở `duong/notebooks/auto_review_queries_gemini38_local.ipynb` trong Jupyter từ repo này. Notebook đọc từng query trong `queries_relational_candidates.csv`, workbook `query_candidates_40.xlsx`, và hai JPEG tương ứng trong `query_images/`. Trước khi chạy, bảo đảm 80 JPEG đã tải về và đặt `GEMINI_API_KEY` trong môi trường hoặc nhập key ẩn khi cell yêu cầu. Notebook tự cài `google-genai` nếu cần.

Cell đầu cho biết số dòng đã có nhãn người và số dòng cần VLM chấm. Đặt `MAX_ROWS=2` để thử; sau đó đặt `MAX_ROWS=None` và chạy lại **cell chấm** trong cùng kernel để xử lý các dòng còn lại. Nếu khởi động lại kernel, điền đường dẫn lượt cũ vào `RUN_DIR` để tiếp tục. Kết quả của mỗi dòng được ghi ngay vào `review/vlm_runs/<lượt>/assessments.jsonl`, kèm hash query và hai ảnh để kiểm tra khi tiếp tục. Cell cuối xuất `queries_validation_vlm_draft.csv` và `queries_validation_vlm_draft.xlsx`, có hai ảnh nhúng, nhận xét từng ảnh, giải thích, độ tin cậy và các trường validation. Workbook gốc và CSV ứng viên được giữ nguyên.

Các dòng đã điền thủ công được giữ theo nhãn người. Nhãn máy có `review_status=vlm_draft` và cần người kiểm tra, sửa rồi chuyển sang `approved` trong bản Excel đầu ra. Sau khi hòa giải, đặt `REVIEWED_WORKBOOK` của notebook 01 tới bản Excel đã duyệt để xuất registry xác minh; không dùng CSV nháp trực tiếp làm ground truth.
