# Audit repo gốc:  @ ``

## 1. Tổng quan

- License: · Ngôn ngữ/framework: · Kích thước: · Cập nhật gần nhất: · Issues đáng chú ý (bug tái lập, số khác paper):
- Cài đặt (đã thử): `pip install -r requirements.txt` ... · Phiên bản ghim được: · Lỗi gặp khi cài và cách sửa:

## 2. Cấu trúc và điểm vào

| Đường dẫn | Vai trò |
|---|---|
| `main.py` / `scripts/*.sh` | |

## 3. Dữ liệu

| Dataset | Tải bằng | Đặt ở | Kích thước | Cần token/đăng ký? |
|---|---|---|---|---|

## 4. Bảng của paper ↔ lệnh gốc ↔ thí nghiệm của mình

| id bảng | Lệnh gốc (đúng tham số) | Đầu ra metric ở đâu (file/log) | exp của mình | Thu nhỏ cho `--limit` bằng cách |
|---|---|---|---|---|

## 5. Giả định ngầm và bẫy

- Đường dẫn cứng / biến môi trường: 
- wandb/hub login, internet, số GPU, bf16/flash-attn (T4 không có bf16, không flash-attn 2):
- Seed có được set không? Chỗ nào ngẫu nhiên không kiểm soát:
- Chỗ code khác với mô tả trong paper:

## 6. Lệnh nhỏ nhất chạy được (đã kiểm chứng, thời gian)

```bash
```

## 7. Khả thi trên compute hiện có

| Thí nghiệm | VRAM/giờ ước tính | T4 16 GB, ≤12 h/kernel | Cách thu nhỏ nếu không vừa |
|---|---|---|---|
