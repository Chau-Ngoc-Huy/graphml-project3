# CLAUDE.md — mmxai

Project 3 (Graph ML): truy vấn văn bản → ảnh, xếp hạng lại bằng đồ thị, có giải thích. Một dự án thống nhất (không chia theo
nhiệm vụ hay theo người). Đọc trước: `README.md`, `docs/PIPELINE.md`, `docs/assignment/Project03_Multimodal_XAI.pdf` (đề),
`docs/data-contract.md`, `results/JOURNAL.md` (lượt mới nhất ở trên).

## Vòng lặp

```
make test                                            # pytest toàn bộ (dữ liệu giả, không cần GPU)
make local EXP=<exp> ARGS="--limit 5"                # máy local; bundle thật: --set data=data/psg_bundle
make push  EXP=<exp> ARGS="--set data=mmxai-vg-bundle --limit 5"   # Kaggle thử; make wait; make pull
make train DS=vg|psg · make adamf DS=.. · make tables · make xai DS=..   # pipeline (docs/PIPELINE.md)
```

Thí nghiệm = `configs/experiments/<exp>.yaml` + `src/mmxai/experiments/<exp>.py` (biến thể chỉ là YAML `extends:` + module 2 dòng
`from .vg_qa import run`).

## Luật (đề, mục 6 — sẽ bị hỏi khi vấn đáp)

- Caption của ảnh val/test không dùng huấn luyện; early stopping và α chọn trên **val**; test chỉ để báo cáo.
- Cạnh thống kê (quan hệ scene graph gộp theo nhãn) chỉ từ ảnh **train** (`build_graph`, có test).
- Ảnh chỉ nhận tin (cạnh nhãn→ảnh) → đặc trưng ảnh val/test không lan sang đỉnh khác (`test_images_never_send`).
- Mọi mô hình dùng chung split và `src/mmxai_eval.py`; ≥ 3 seed cho lượt chính, báo mean ± std.
- Giải thích phải đến từ chính cơ chế tính điểm (`graph/explain.py`, `xai_bridge.py` chạy lại mô hình trên đồ thị đã xóa cạnh)
  và phải qua kiểm tra fidelity.
- Không commit `data/`, `runs/`, `models/`, `cache/`, `build/`, `local.mk`, `.env`, token.
- Trên CPU giới hạn `cpu_threads` (2–4): nhiều luồng chậm hơn với đồ thị nhỏ; chạy nhẹ, tuần tự (`nice -n 19`).
