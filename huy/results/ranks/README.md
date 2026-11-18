# Hạng của ảnh đúng cho từng truy vấn (toàn pool test, 2138 ảnh, 10 696 caption), seed 0

Cùng thứ tự truy vấn với `ranks_clip_t2i_test.npz["full"]` của Thắng và `protocol_coco_full.json` của Khánh → dùng thẳng cho
kiểm định ghép cặp (bootstrap theo truy vấn). Mỗi file: `{"seed": 0, "ranks": [...]}`, hạng bắt đầu từ 1.
Nguồn: `runs/<exp>/*kaggle-3seed*/ranks_test_full.json` (Kaggle T4, JOURNAL #9). Cấu hình chính đề xuất: `t1_qa_cn_no_sg_sage`.
