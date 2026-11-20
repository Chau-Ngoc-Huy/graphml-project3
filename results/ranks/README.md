# Hạng của ảnh đúng cho từng truy vấn (toàn pool test, 2138 ảnh, 10 696 caption), seed 0

Cùng thứ tự truy vấn với `ranks_clip_t2i_test.npz["full"]` của Thắng và `protocol_coco_full.json` của Khánh → dùng thẳng cho
kiểm định ghép cặp (bootstrap theo truy vấn). Mỗi file: `{"seed": 0, "ranks": [...]}`, hạng bắt đầu từ 1.
Nguồn: `runs/<exp>/*kaggle-3seed*/ranks_test_full.json` (Kaggle T4, JOURNAL #9). Cấu hình chính đề xuất: `vg_qa_cn_no_sg_sage`.

## PSG (`psg_*`, 2186 ảnh test, 10 933 caption)

`psg_*.seed0.test_full.json`: cùng định dạng, thứ tự truy vấn của giao thức trên `data/psg_bundle` (dataset Kaggle
`huychau/mmxai-psg-bundle`). Nguồn: `runs/psg_*/*-s0*/ranks_test_full.json` (máy local, JOURNAL #10). Hạng của AdaMF-MAT /
CLIP / khớp tên trên PSG: `results/adamf_psg/results_PSG_*.json` (`[<phương pháp>]["full"]["ranks"]`, JOURNAL #11).
