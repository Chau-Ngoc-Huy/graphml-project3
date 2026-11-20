# yen/

Folder dữ liệu Visual Genome, chuẩn hóa nhãn và split: dùng split benchmark COCO-ID đã có: train 15.000, val 2.126, test 2.138. Dữ liệu lớn nằm trong `yen/data/`.

## VG -> MMXAI

1. Tải `scene_graphs.json` và `image_data.json` của Visual Genome; tải `dataset_coco.json`
   (Karpathy) để có caption COCO. Giải nén ảnh VG sao cho tên file là `<VG image_id>.jpg`.
2. Tạo bundle theo hợp đồng `../shared/data-contract.md`:

   ```powershell
   python yen\prepare_visual_genome.py `
     --scene-graphs <path>\scene_graphs.json `
     --image-data <path>\image_data.json `
     --coco-captions <path>\dataset_coco.json `
     --output yen\data\visual_genome_mmxai
   ```

   Mặc định là split Karpathy/COCO-ID (train 15.000, val 2.126, test 2.138); script
   ánh xạ qua `image_data.json` sang VG image ID và dừng nếu thiếu ảnh/annotation
   cần thiết. Khi cần thí nghiệm độc lập, dùng `--random-split --seed 0
   --ratios 0.8 0.1 0.1`; `splits.json` đã sinh là tập cố định cho các lần chạy sau.

3. Trích CLIP vào `features/` (cần `open_clip_torch`, `pillow`, `numpy`, `torch`):

   ```powershell
   python yen\extract_visual_genome_features.py `
     --data yen\data\visual_genome_mmxai `
     --images-dir <path>\VG_100K <path>\VG_100K_2
   ```

   Kết quả gồm `image_emb.npy`/`image_ids.json`, `caption_emb.npy`/`caption_ids.json`,
   `label_emb.npy`/`labels.json` và `clip_model.json`. Tất cả embedding được
   L2-normalize; vocab nhãn và label embedding chỉ lấy từ ảnh train.

Scene graph có thể thiếu object nhưng quan hệ còn lưu tên endpoint. Bộ tiền xử lý
bổ sung object đó vào `vg/objects.json`; quan hệ có endpoint không định danh được
không bị đoán mà được liệt kê trong `preprocess_report.json`. Ví dụ `1592647` được kiểm tra bằng test: object có tên
endpoint trong quan hệ được phục hồi; endpoint chỉ có ID sẽ được báo là chưa giải
quyết. Báo cáo thực tế chỉ có mục cho ảnh trong split được chọn. Nếu quan hệ chỉ có
ID và object gốc không tồn tại, cần sửa/tìm lại annotation nguồn trước khi dùng nó
làm bằng chứng.

## Split và chống rò rỉ

- Split benchmark dùng COCO ID, split nhóm dùng VG image ID. Không chép
  COCO ID trực tiếp vào `splits.json`; script ánh xạ bằng `coco_id` và giữ mọi VG ID
  tương ứng với cùng COCO ảnh trong một split.
- `train`, `val`, `test` được kiểm tra không giao nhau. Dùng duy nhất cùng
  `splits.json` cho CLIP, CLIP+Graph và các mô hình khác; không chia caption độc lập
  theo từng câu.
- Vocab nhãn cho text embedding chỉ học từ train. Chọn tham số/early stopping trên
  val; test chỉ dùng cho đánh giá cuối. Không tính thống kê từ test caption/object,
  không đưa caption hoặc quan hệ của test vào loss/đồ thị train. Feature CLIP ảnh
  đóng băng có thể tính cho mọi split, nhưng không fine-tune encoder trên val/test.
- Object/relationship scene graph của test chỉ được dùng như input tại suy luận nếu
  tác vụ cho phép scene graph; không dùng quan hệ test làm nhãn huấn luyện.
- `--random-split` dùng seed cố định, nhóm theo COCO ID trước khi chia; file
  `splits.json` được lưu để tái sử dụng, không tạo lại split riêng cho mỗi model.

Kiểm tra / chạy MMXAI theo hướng dẫn tại `../shared/data-contract.md` và
`../huy/README.md`; `--set data=yen/data/visual_genome_mmxai`.
