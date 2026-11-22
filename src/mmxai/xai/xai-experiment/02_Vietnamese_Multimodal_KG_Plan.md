# Plan 2 — Xây dựng Multimodal Knowledge Graph (MMKG) tiếng Việt bằng pipeline tự động

## 1. Mục tiêu và giới hạn

- [ ] Tự xây dựng MMKG quy mô nhỏ trong **một miền hẹp**, ưu tiên **Ẩm thực Việt Nam**.
- [ ] Quy mô mục tiêu: **~100 thực thể/khái niệm, ~500–700 triple quan hệ, ~300 ảnh**, thuộc phạm vi gợi ý của đề (**50–150 thực thể, 300–1.000 triple, 200–500 ảnh**).
- [ ] Ưu tiên **thu thập dữ liệu có cấu trúc → trích xuất tự động từ văn bản → entity linking tự động → kiểm tra tự động** thay vì annotation toàn bộ bằng tay.
- [ ] Liên kết ảnh với thực thể/khái niệm bằng bằng chứng từ metadata, caption hoặc model-assisted prediction có lọc.
- [ ] Tạo data card đầy đủ nguồn dữ liệu, cách sinh triple, quy tắc lọc, bản quyền và hạn chế.
- [ ] Chạy CLIP-only và retrieval có graph trên MMKG mới; đánh giá transfer và một số giải thích từ Plan 1.

**Ngoài phạm vi:** thu thập ảnh tùy ý từ web không rõ giấy phép, annotate thủ công toàn bộ, xây ontology phức tạp, fine-tune VLM từ đầu, triển khai graph database nặng.

## 2. Chốt schema trước khi thu thập

- [ ] Chọn các loại node tối thiểu:
  - `Image`: `IMG_0001`, `IMG_0002`, ...
  - `Entity`: món ăn, nguyên liệu, đối tượng quan sát được.
  - `Concept`: loại món, vùng ẩm thực, cách chế biến.
- [ ] Chọn quan hệ hữu hạn: `depicts`, `is_a`, `has_ingredient`, `served_with`, `originates_from` (chỉ giữ relation có đủ dữ liệu).
- [ ] Định nghĩa rõ **triple tri thức** (`entity-relation-entity`) và **liên kết ảnh–thực thể** (`image-depicts-entity`) là hai bảng riêng; báo cáo số lượng của từng loại, tránh đếm lẫn.
- [ ] Định nghĩa format chuẩn:

```text
entities.csv: entity_id, label_vi, label_en, entity_type, source_id, aliases
triples.csv: head_id, relation, tail_id, source_id, extraction_method, confidence, evidence_text
images.csv: image_id, local_path, source_url, description_vi, creator, license, license_url, original_id
image_entity_links.csv: image_id, entity_id, relation, evidence_source, extraction_method, confidence
```

- [ ] Tạo `schema.json`/`vocabulary.json` quy định cặp node-type hợp lệ cho từng relation.
- [ ] Lưu `source_id`, URL/ID nguồn và phương pháp trích xuất để truy ngược mỗi triple/link.

## 3. Thu thập ảnh, thực thể và triple có cấu trúc

### 3.1. Nguồn ưu tiên

- [ ] Truy xuất thực thể và quan hệ từ **Wikidata** (SPARQL/API), ưu tiên nhãn tiếng Việt; giữ ID Wikidata để gộp entity.
- [ ] Lấy ảnh có liên quan từ **Wikimedia Commons**, ưu tiên ảnh có đối tượng được mô tả rõ; lưu metadata bản quyền **theo từng file ảnh**.
- [ ] Nếu thiếu ảnh/caption tiếng Việt, xem xét **UIT-OpenViIC** hoặc nguồn ảnh–caption phù hợp **sau khi kiểm tra điều khoản sử dụng và quyền phân phối**.
- [ ] Không dựa vào tên file hay kết quả tìm kiếm ảnh làm ground truth cho `depicts` khi không có bằng chứng hỗ trợ.

### 3.2. Bộ thu thập tự động

- [ ] Viết `collect_entities.py`: tải danh sách thực thể thuộc miền, nhãn, alias và quan hệ có cấu trúc.
- [ ] Viết `collect_images.py`: tải ảnh và metadata (`source`, `creator`, `license`, `license_url`, `description`, `depicts` nếu có).
- [ ] Thiết lập giới hạn truy vấn/API, retry và bộ nhớ đệm để chạy lại mà không tải trùng.
- [ ] Loại ảnh lỗi, thumbnail, quá nhỏ, ảnh trùng theo hash/perceptual hash.
- [ ] Loại ảnh thiếu bằng chứng về quyền sử dụng; không suy đoán license từ nguồn chung.
- [ ] Không đưa dữ liệu cá nhân nhạy cảm không cần thiết vào MMKG.

## 4. Tạo triple tự động và chuẩn hóa tiếng Việt

### 4.1. Ưu tiên triple có nguồn cấu trúc

- [ ] Chuyển các quan hệ Wikidata hợp lệ sang vocabulary nội bộ.
- [ ] Không tự ép quan hệ nguồn khác nghĩa thành `has_ingredient`, `depicts` hoặc `served_with`.
- [ ] Chỉ giữ triple có head/tail trong danh sách entity hoặc có quy trình mở rộng entity rõ ràng.

### 4.2. Trích xuất có kiểm soát từ văn bản

- [ ] Chuẩn hóa description/caption về tiếng Việt hoặc giữ song ngữ kèm mã ngôn ngữ.
- [ ] Dùng rule-based patterns cho các quan hệ dễ trích xuất (`là một loại`, `có nguyên liệu`, `ăn kèm`).
- [ ] Dùng LLM theo **schema cố định** để trích xuất triple từ **văn bản nguồn**, không yêu cầu sinh KG từ trí nhớ.
- [ ] Yêu cầu LLM trả `head`, `relation`, `tail`, `evidence_span`; bỏ triple không tìm được bằng chứng trong input.
- [ ] Chuẩn hóa dấu, chữ hoa/thường, dấu câu, alias tiếng Việt; map `phở bò`/biến thể về một `entity_id` khi có căn cứ.
- [ ] Chặn triple tự mâu thuẫn, sai kiểu quan hệ hoặc trùng lặp; với nguồn khác nhau, giữ nhiều evidence/provenance thay vì tạo nhiều triple giống nhau.
- [ ] Lưu phương pháp và ngưỡng confidence; không gán `confidence` tùy ý như một xác suất đã hiệu chuẩn.

## 5. Liên kết ảnh–thực thể tự động

- [ ] Thực hiện theo thứ tự ưu tiên bằng chứng:
  1. **Structured image metadata** liên kết ảnh với thực thể.
  2. **Caption/description entity linking** (exact/fuzzy aliases + lọc ngữ cảnh).
  3. **Model-assisted labeling** bằng VLM/image–text similarity cho các ảnh còn thiếu liên kết.
- [ ] Kiểm tra model predictions bằng ngưỡng được chọn trên validation/mẫu kiểm tra độc lập; lưu trạng thái `auto_verified` hoặc `needs_review`.
- [ ] Không suy từ triple `món ăn has_ingredient X` thành `image depicts X` nếu trong ảnh không có bằng chứng X xuất hiện.
- [ ] Mỗi ảnh được giữ lại phải có ≥1 liên kết `image_entity_links` có căn cứ; lưu `evidence_source`.
- [ ] Loại/đưa vào danh sách kiểm tra các ảnh mà metadata, caption và model có nhãn mâu thuẫn.

## 6. Kiểm định chất lượng và chống rò rỉ

### 6.1. Tự động hóa kiểm định

- [ ] Kiểm tra duplicate ảnh, entity, triple, image-entity link.
- [ ] Kiểm tra node tồn tại, quan hệ hợp lệ theo schema, `source_id` không rỗng, license metadata đầy đủ.
- [ ] Báo thống kê theo relation, nguồn, số ảnh/entity, degree distribution, orphan nodes, thành phần liên thông.
- [ ] Báo tỷ lệ ảnh có caption tiếng Việt, ảnh có liên kết entity và triple có evidence.
- [ ] Kiểm tra phân phối class để phát hiện dataset quá lệch về vài món ăn.

### 6.2. Chỉ kiểm tra thủ công theo mẫu, không annotate toàn tập

- [ ] Lấy mẫu ngẫu nhiên khoảng **30 ảnh** và **50 triple**; bổ sung một mẫu nhỏ các trường hợp low-confidence.
- [ ] Ghi `correct`/`incorrect`/`uncertain` và nguyên nhân; báo precision ước lượng cùng giới hạn của mẫu.
- [ ] Nếu chất lượng không đạt, sửa **rule/threshold/prompt chung** rồi chạy lại pipeline; không sửa thủ công từng hàng trừ ngoại lệ được ghi nhận.

### 6.3. Split và leakage

- [ ] Cố định seed và train/validation/test image IDs; ảnh giống/near-duplicate phải vào cùng split.
- [ ] Không dùng ảnh/caption test để học tham số, chọn ngưỡng, xây **cạnh thống kê đồng xuất hiện** hoặc tăng cường dữ liệu train.
- [ ] Có thể sử dụng **tri thức KG ngoài độc lập với benchmark** nếu ghi rõ nguồn và quy tắc; phân biệt với cạnh suy từ tập ảnh đang đánh giá.
- [ ] Tách rõ `transductive`/`inductive` hoặc `zero-shot`/`adaptation` trong thí nghiệm chuyển giao; không tuyên bố zero-shot nếu có huấn luyện trên dữ liệu mới.

## 7. Bộ truy vấn text-to-image tiếng Việt và đánh giá chuyển giao

- [ ] Tạo **≥20 truy vấn tiếng Việt**, chia tối thiểu thành **direct** và **indirect**.
- [ ] Ví dụ direct: `Một tô phở bò trên bàn ăn`.
- [ ] Ví dụ indirect: `Món nước có bánh phở và thịt bò`.
- [ ] Với mỗi truy vấn, xác lập ảnh đúng bằng evidence từ metadata/caption/triple và rà soát mẫu; **không lấy nhãn relevance từ score của mô hình**.
- [ ] Cài cùng tập ứng viên và cùng split cho mọi phương pháp:
  - `CLIP-only` (hoặc encoder ảnh–văn bản đa ngôn ngữ phù hợp).
  - `CLIP + Graph` bằng scorer/model tốt nhất của nhóm.
  - `CLIP + Graph + xAI` dùng module giải thích của Plan 1.
- [ ] Nếu dùng bản dịch truy vấn tiếng Việt → tiếng Anh để tương thích CLIP, lưu bản dịch và dùng cùng một chiến lược cho tất cả baseline; báo đây là phương án dịch, không phải native Vietnamese embedding.
- [ ] Báo `Recall@1/5/10`, `MRR`; tách direct vs indirect và phân tích sự thay đổi khi chuyển sang miền/tiếng Việt.
- [ ] Đối với GNN huấn luyện trên graph khác ID/schema: kiểm tra có thể **transfer trực tiếp** hay phải tạo embedding/huấn luyện thêm; nếu cần thích nghi, chỉ dùng train/validation của MMKG và ghi rõ protocol.
- [ ] Chạy giải thích cho ≥10 truy vấn theo Plan 1; đối chiếu cạnh explanation với provenance của MMKG mới.
- [ ] Với thực nghiệm chính có học tham số, báo trung bình ± độ lệch chuẩn trên **ít nhất 3 seed**; không chọn phương pháp dựa trên tập test.

## 8. Đầu ra và data card

- [ ] `mmkg_vi/collect_entities.py`, `collect_images.py`.
- [ ] `mmkg_vi/extract_triples.py`, `entity_linking.py`, `validate_graph.py`.
- [ ] `mmkg_vi/data/entities.csv`.
- [ ] `mmkg_vi/data/triples.csv`.
- [ ] `mmkg_vi/data/images.csv`.
- [ ] `mmkg_vi/data/image_entity_links.csv`.
- [ ] `mmkg_vi/data/splits.json`, `query_eval_vi.csv`.
- [ ] `mmkg_vi/data/data_card.md`, gồm:
  - Phạm vi miền và mục đích sử dụng.
  - Nguồn, giấy phép và điều kiện dùng/phân phối ảnh–văn bản.
  - Node/relation schema, số lượng thực tế, ngôn ngữ và class distribution.
  - Cách thu thập, tạo triple, entity linking, lọc và kiểm tra chất lượng.
  - Provenance, nguy cơ nhiễu nhãn, bias, thiếu coverage và hạn chế.
  - Cách chia dữ liệu, hạn chế rò rỉ và cách tái lập.
- [ ] `experiments/transfer_results.csv`, biểu đồ CLIP-only vs CLIP+Graph, direct vs indirect.
- [ ] Notebook nhập truy vấn tiếng Việt, trả top-k ảnh và đường đi/đồ thị con giải thích.
- [ ] README có dependencies, cấu hình nguồn dữ liệu, lệnh chạy toàn pipeline và ví dụ input/output.

### Tiêu chí nghiệm thu

- [ ] MMKG có **≥200 ảnh**, **50–150 thực thể/khái niệm**, **≥300 triple tri thức** (báo riêng các cạnh image–entity) nếu nguồn dữ liệu hợp lệ cho phép đạt quy mô này.
- [ ] Truy xuất được nguồn/bằng chứng cho mỗi triple và mỗi image–entity link được công bố.
- [ ] Các ảnh được sử dụng có metadata bản quyền và điều kiện sử dụng rõ ràng.
- [ ] Có bộ test text-to-image tiếng Việt và so sánh CLIP-only vs CLIP+Graph trên **cùng candidate pool**.
- [ ] Có phân tích transfer và ít nhất 10 ví dụ giải thích, bao gồm các trường hợp lỗi.
- [ ] Pipeline có thể tái chạy mà không cần annotation thủ công toàn bộ dataset.

## 9. Phương án tối giản nếu nguồn dữ liệu thiếu

- [ ] Thu hẹp số lớp món ăn nhưng giữ độ đa dạng ảnh và quan hệ.
- [ ] Ưu tiên triple có nguồn rõ ràng hơn triple LLM suy đoán.
- [ ] Giữ schema đơn giản (`depicts`, `is_a`, `has_ingredient`) nếu các quan hệ khác quá thưa.
- [ ] Nếu không thu đủ ảnh có license hợp lệ, báo rõ mức thực tế và giới hạn; **không** tạo số lượng cho đủ bằng ảnh trùng/ảnh không rõ quyền.
- [ ] Giữ khả năng xuất được `entities`, `triples`, `images`, `image_entity_links`, query set và thí nghiệm retrieval/xAI ngay cả khi không dùng graph database.
