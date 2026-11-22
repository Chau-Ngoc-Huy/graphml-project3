# Kết quả khảo sát XAI trên VG và PSG

Thư mục này lưu hai lượt đánh giá được thực hiện ngày 21/11/2026. Tệp `dual-20261121-125423_manifest.json` ghi thông tin về các mô hình dùng chung. Mỗi thư mục lượt chạy giữ nguyên tên và cấu trúc kết quả để đối chiếu với dữ liệu phân tích.

- `xai-20261121-130842/`: khảo sát giải thích ở mức cạnh trên tập truy vấn được chọn. Các bảng tổng hợp nằm trong `analysis/`; `analysis/xai_results_corrected.csv` là bảng kết quả đã hiệu chỉnh được dùng cho phân tích cuối cùng. Tệp `xai_results.csv` lưu kết quả ban đầu để đối chiếu.
- `path-20261121-184857/`: khảo sát truy vấn trực tiếp và đường giải thích. `case_manifest.csv`, `xai_results.csv` và `paths.jsonl` lưu dữ liệu từng trường hợp; `analysis/` chứa các bảng tổng hợp, khoảng tin cậy và `validity_case_review_10.xlsx` để rà soát 10 trường hợp minh họa.

Hai lượt chạy dùng tập truy vấn và giao thức đánh giá khác nhau; không gộp trực tiếp các chỉ số của chúng thành một ước lượng chung. Các ảnh trong `analysis/case_images/` hỗ trợ kiểm tra trực quan các trường hợp được chọn. Các bản nháp báo cáo và tệp hệ thống không thuộc bản lưu kết quả này.

Mã, notebook và toàn bộ thư mục chạy gốc nằm trong `src/mmxai/xai/xai-experiment/`. Thư mục này gồm checkpoint của sáu seed và hai lượt kết quả; bản tại `results/xai/qa_vg_psg_xai/` giữ các artifacts cần đối chiếu báo cáo. Xem `docs/PIPELINE.md` để biết thứ tự chạy và tệp dùng đối chiếu Bảng X, Hình 6–7.
