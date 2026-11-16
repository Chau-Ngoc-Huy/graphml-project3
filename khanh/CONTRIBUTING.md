# Quy ước làm việc (khanh/)

- Chỉ sửa trong `khanh/`. Cần đổi `shared/`, `thang/`, `huy/`: nhắn nhóm hoặc làm Pull Request.
- Không commit `data/`, `runs/`, `checkpoint/`, `external/`, `.venv/`, `.env`, token, ảnh, `.npy/.npz/.pth`. Dữ liệu và checkpoint để trên Kaggle.
- Sửa mã trong `src/`, rồi `make notebooks` để sinh lại notebook; notebook nhúng nguyên mã nguồn nên không sửa tay.
- Mỗi lượt chạy Kaggle: ghi một đoạn vào `results/JOURNAL.md` (ngày, link phiên bản Kaggle, cấu hình, số chính, hạn chế) và commit `results/**/*.json`.
- Nhánh `exp/khanh/<việc>`, PR vào `main`, hoặc push thẳng vào `khanh/` (không ảnh hưởng người khác). Trước khi push: `git pull --rebase`.
- Commit nhỏ, message dạng `khanh: <làm gì>`.
