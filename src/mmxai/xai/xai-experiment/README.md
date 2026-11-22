# Bộ thí nghiệm XAI của Dương

Thư mục này giữ bộ thí nghiệm XAI tách biệt, lấy từ `duong/` trên nhánh `main`. Gói Python ở đây có tên `xai_experiment`; gói `mmxai.xai` của nhóm ở thư mục cha vẫn là thành phần riêng. Mã, notebooks, tests, cấu hình, registry truy vấn và workbook đã duyệt được giữ cùng nhau. Báo cáo và các lượt chạy lịch sử vẫn ở `main`, không sao chép vào đây.

Các lượt chạy mới dùng pipeline hiện tại của `final`, vì vậy kết quả có thể khác số liệu lịch sử. Ảnh, bundle VG/PSG và checkpoint không nằm trong thư mục này.

## Thiết lập

Dùng Python 3.10 trở lên. Từ gốc repository:

```bash
python -m pip install -e '.[viz]'
python -m pip install -e 'src/mmxai/xai/xai-experiment[notebooks]'
```

Trên Colab, đặt `graphml-project3/`, `mmxai_vg_bundle/` và `mmxai_psg_bundle/` cùng cấp trong MyDrive. Notebook 01–03 dùng Colab; notebook 04–06 cũng có thể chạy cục bộ sau khi đã cấp bundle và checkpoint cần thiết. Chạy 01 đến 06 theo thứ tự: huấn luyện, can thiệp baseline, phân tích baseline, audit truy vấn, can thiệp direct và phân tích direct. Notebook Gemini chỉ tạo nhãn nháp, không tự ghi đè nhãn người trong workbook.

Output mặc định nằm ở `runs/qa_vg_psg_xai/` trong thư mục này và bị Git bỏ qua. Có thể đặt biến môi trường `XAI_EXPERIMENT_RUNS_DIR` để dùng một thư mục ngoài repository. Notebook tự chọn manifest hoặc snapshot hoàn tất mới nhất; các ô thiết lập cho phép thay bằng đường dẫn cụ thể. CSV trong `registry/` và các file SHA-256 đi kèm là đầu vào có phiên bản. Workbook trong `review/` chứa nhãn đã duyệt.

## Kiểm tra

Từ thư mục này:

```bash
python -m unittest discover -s tests -v
```

Các unit test dùng dữ liệu mẫu nhỏ. Chạy đủ VG/PSG trên GPU cần bundle ngoài repository; không dùng unit test để suy diễn lại số liệu trong báo cáo.
