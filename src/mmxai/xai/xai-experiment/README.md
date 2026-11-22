# XAI cho truy hồi văn bản–ảnh

Thư mục này chứa hai lượt đánh giá giải thích bằng can thiệp cạnh đồ thị trên cấu hình Visual Genome (VG) và Panoptic Scene Graph (PSG). [Báo cáo tổng kết](reports/XAI_Final_Report.md) trình bày kết quả, hình trực quan và giới hạn của thí nghiệm. [Kế hoạch XAI](01_XAI_Explainability_Plan.md) ghi thiết kế và tiêu chí đo chi tiết.

## Trạng thái kết quả

- **Khảo sát ban đầu:** 24 câu mô tả ảnh, ba phiên bản mô hình, ngân sách 1/3/5 cạnh. Snapshot checkpoint và kết quả đã được lưu trong `runs/qa_vg_psg_xai/`.
- **Lượt direct:** 40 ứng viên được duyệt trong một workbook; 34 truy vấn direct có ảnh chuẩn hỗ trợ được đưa vào can thiệp cạnh đối tượng. Notebook phân tích chứa output của lượt đã chạy; artifact đầy đủ của lượt này được lưu trên Drive cùng dự án.
- **Kiểm tra ảnh:** 10 trường hợp minh họa được chọn cân bằng theo khớp/không khớp ID ảnh chuẩn. Cả năm ảnh khác ID trong mẫu này được người dùng xác nhận hỗ trợ truy vấn. Mẫu này không ước lượng độ chính xác ngữ nghĩa của toàn bộ 34 truy vấn.

## Notebook theo thứ tự

| Bước | Notebook | Môi trường | Mục đích |
|---:|---|---|---|
| 01 | [Huấn luyện VG/PSG](notebooks/01_train_vg_psg_models_colab.ipynb) | Colab GPU | Huấn luyện ba phiên bản mỗi cấu hình, lưu checkpoint và hạng gốc. Chỉ cần chạy lại khi tái tạo baseline từ đầu. |
| 02 | [Can thiệp XAI ban đầu](notebooks/02_baseline_xai_colab.ipynb) | Colab GPU | Xóa cạnh giải thích hoặc cạnh ngẫu nhiên, đo thay đổi điểm/hạng trên 24 câu mô tả. |
| 03 | [Phân tích baseline](notebooks/03_baseline_analysis_colab.ipynb) | Colab CPU | Tổng hợp 216 lượt, bootstrap theo câu mô tả, tạo bảng và hình. |
| 04 | [Audit truy vấn direct](notebooks/04_direct_query_audit.ipynb) | CPU/Colab | Đọc sheet tổng, kiểm tra 40 ứng viên và xuất registry 34 truy vấn đủ điều kiện cùng lý do loại. |
| 05 | [Can thiệp cạnh đối tượng](notebooks/05_direct_object_xai_colab.ipynb) | Colab GPU | Xóa cạnh `contains` khớp truy vấn, chạy lại scorer trên cùng top-100, so với cạnh khác của cùng ảnh. |
| 06 | [Phân tích direct và kiểm tra ảnh](notebooks/06_direct_xai_analysis.ipynb) | CPU/Colab | Báo coverage, fidelity, sparsity, CI theo truy vấn, khớp ID và mức phù hợp ngữ nghĩa của mẫu ảnh. |

[Notebook Gemini tùy chọn](notebooks/optional_gemini_review_local.ipynb) từng dùng để gợi ý nhãn trước khi duyệt. Cả 40 dòng trong workbook hiện đã được duyệt, nên notebook này **không thuộc chuỗi chạy chính**. Nó tự trích ảnh đã nhúng trong workbook ra bộ nhớ đệm cục bộ khi cần dùng lại; bản nháp VLM không tự thay nhãn người.

`notebooks/build_path_notebooks.py` tạo lại **bản không có output** của ba notebook 04–06. Chạy script này sẽ ghi đè output đã lưu trong các notebook đó; chỉ dùng khi chủ động tạo lại notebook nguồn.

## Dữ liệu và provenance

- [Sheet tổng đã duyệt](review/query_candidates_40.xlsx) là **file duy nhất trong `review/`**. Workbook chứa 40 dòng và ảnh GT/ảnh đối chứng nhúng, nên không cần một thư mục JPEG đi kèm.
- `registry/` giữ danh sách ứng viên khóa trước thí nghiệm, registry đã nhập nhãn từ workbook, checksum và bảng lý do đủ/không đủ điều kiện. Đây là dữ liệu máy đọc cho các notebook 04–06; notebook 04 có thể tạo lại registry đã duyệt nếu workbook đổi.
- `runs/qa_vg_psg_xai/` chứa snapshot baseline đã commit. Lượt direct mới có thể sinh thư mục `path-*`; các lượt mới không được tự động thêm vào Git.
- `reports/` chứa báo cáo khoa học và hai hình. Các file cục bộ phục vụ tải ảnh hoặc VLM nằm trong `.local/` và bị Git bỏ qua.

Các bundle VG/PSG đầu vào nằm ngoài repo. Trên Drive, đặt `graphml-project3/`, `mmxai_vg_bundle/` và `mmxai_psg_bundle/` cùng cấp để chạy lại notebook GPU. Notebook direct kiểm tra hash checkpoint và bundle trước khi can thiệp. Điểm fidelity đo thay đổi của mô hình trên 100 ảnh ứng viên cố định; việc ảnh khác ID GT vẫn phù hợp truy vấn được chấm riêng.

## Kiểm tra mã

Từ `duong/`:

```bash
python -m unittest discover -s tests -v
```

Các kiểm tra lõi dùng dữ liệu mẫu nhỏ; việc chạy lại thí nghiệm GPU cần bundle ngoài repo và checkpoint đã lưu. Xem [báo cáo tổng kết](reports/XAI_Final_Report.md) để đọc kết quả đã xác minh.
