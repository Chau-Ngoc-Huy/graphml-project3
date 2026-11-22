# Tổng hợp XAI VG và PSG

Nguồn: lượt `xai-20261121-130842` trên Colab, tải về tại thư mục cha của báo cáo này; 24 caption, 3 model seed, k=1/3/5; 216 dòng. Random baseline: 20 lượt cho mỗi dòng, khớp số lượng và loại cạnh. Số trong các bảng được làm tròn đến ba chữ số thập phân.

## Câu hỏi, thiết kế và cách đọc metric

Thí nghiệm kiểm tra liệu các cạnh do XAI chọn có thực sự ảnh hưởng đến **điểm của ảnh mà mô hình xếp hạng 1** hay không. Với mỗi truy vấn, mô hình giữ nguyên 100 ảnh ứng viên. Explainer đo mức giảm điểm khi xóa riêng từng cạnh trong vùng được sàng lọc, chọn tối đa k cạnh có đóng góp dương, rồi xóa **đồng thời** các cạnh đã chọn và chạy lại scorer. Đối chứng xóa cùng số cạnh được lấy ngẫu nhiên từ cùng vùng, khớp số cạnh theo loại (`contains`, `concept`, `sg_rel`), lặp 20 lần. [Mã explainer](../../../../xai/explain.py), [mã evaluator](../../../../xai/evaluate_xai.py) và [notebook XAI](../../../../notebooks/xai_vg_psg_colab.ipynb) xác định phép đo này.

- `final_drop` = điểm fuse của ảnh top-1 trước can thiệp trừ điểm sau can thiệp; `random_drop` là trung bình 20 phép xóa ngẫu nhiên của **cùng truy vấn, cùng model seed và cùng k**. Điểm fuse dùng thang z-score, nên các giá trị này là mức đổi điểm, không phải phần trăm hay mức đổi MRR.
- `advantage` = `final_drop − random_drop`. Giá trị dương cho thấy tập cạnh XAI làm giảm điểm ảnh đang được giải thích nhiều hơn đối chứng. Đây là fidelity theo phép xóa, không xác nhận cạnh phản ánh nội dung ảnh một cách đúng đắn.
- `top1_changed` là tỷ lệ ảnh top-1 ban đầu tụt khỏi hạng 1; `gt_improved` là tỷ lệ **ảnh đúng** có hạng tốt hơn sau khi xóa cạnh. Hai metric hỏi hai điều khác nhau. `coverage_pct` là phần trăm cạnh trong vùng 2-hop của ảnh top-1 được đưa vào tập cạnh sàng lọc, không phải độ bao phủ toàn bộ graph.

Mỗi dataset chọn 3 caption cho từng ô `direct/indirect × success/failure` theo **outcome của seed 0**: success có ảnh đúng ở hạng 1, failure có ảnh đúng ở hạng 11–100; hạng 2–10 không được chọn. Cùng 12 caption được đánh giá trên cả ba model seed, nên mỗi hàng tổng thể ở một mức k có 36 quan sát. `Direct` nghĩa là mọi khái niệm trích được từ caption (sau khi bỏ từ chung) đều có trong nhãn đối tượng của ảnh đúng; `indirect` nghĩa là có ít nhất một khái niệm thiếu trong các nhãn đó. Đây là nhãn dữ liệu do [bộ đánh giá của Thắng](../../../../../thang/mmxai_eval.py) tạo, không bảo đảm mô hình đã suy luận qua đường đi nhiều bước. [Danh sách case](../selected_cases.json), [CSV đã sửa seed](xai_results_corrected.csv), [summary gốc](../summary.json) và [manifest checkpoint](../../dual-20261121-125423_manifest.json) cho phép đối chiếu thiết kế.

Checkpoint gốc có full test MRR lần lượt **65,092 / 64,803 / 64,762** cho VG và **64,424 / 64,352 / 64,184** cho PSG (seed 0/1/2). Đây là chất lượng truy vấn trước can thiệp, chỉ dùng làm bối cảnh; MRR trên toàn tập test không được tính lại trong thí nghiệm XAI trên 24 caption đã chọn. Xem [training_baseline.csv](training_baseline.csv).

## Tổng thể

| dataset | requested_k | n | final_drop | random_drop | advantage | top1_changed | gt_improved | coverage_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| psg | 1 | 36 | 0.901 | 0.178 | 0.723 | 0.778 | 0.278 | 0.477 |
| psg | 3 | 36 | 1.723 | 0.726 | 0.996 | 0.944 | 0.500 | 0.477 |
| psg | 5 | 36 | 1.726 | 1.012 | 0.715 | 0.889 | 0.472 | 0.477 |
| vg | 1 | 36 | 0.490 | 0.050 | 0.440 | 0.694 | 0.139 | 0.869 |
| vg | 3 | 36 | 0.902 | 0.173 | 0.729 | 0.806 | 0.194 | 0.869 |
| vg | 5 | 36 | 1.062 | 0.220 | 0.843 | 0.833 | 0.278 | 0.869 |

![Giảm điểm fuse của ảnh top-1 khi xóa cạnh XAI và cạnh ngẫu nhiên cùng loại, theo k trên VG và PSG](fidelity_vs_random.png)

*Hình 1.* Cột xanh là `final_drop`, cột cam là `random_drop`; mỗi cặp cột tổng hợp 12 caption × 3 model seed. [Giá trị chưa làm tròn](overall.csv).

Ở cả sáu tổ hợp dataset–k, mức giảm điểm sau xóa cạnh XAI cao hơn random baseline. Chẳng hạn tại k=3, VG giảm **0,902** so với **0,173** (advantage **0,729**), PSG giảm **1,723** so với **0,726** (advantage **0,996**). Tăng k từ 1 lên 3 thường tăng mức giảm tuyệt đối, song tăng tiếp lên 5 cho lợi ích thêm ít hơn: PSG gần như giữ nguyên `final_drop` (**1,723 → 1,726**) trong khi `random_drop` tăng (**0,726 → 1,012**), khiến advantage giảm từ **0,996** xuống **0,715**. Đây là đặc điểm của tập case và scorer hiện tại, không đủ để kết luận k=3 tối ưu chung.

Ở k=3, ảnh top-1 ban đầu rời hạng 1 trong **29/36** lượt VG và **34/36** lượt PSG. Ảnh đúng chỉ tăng hạng trong **7/36** lượt VG và **18/36** lượt PSG; lần lượt **7/36** và **14/36** lượt còn bị giảm hạng. XAI đang giải thích sự phụ thuộc của **dự đoán hiện tại** vào cạnh, nên điểm ảnh top-1 giảm là bằng chứng về tác động của cạnh, còn ảnh đúng có thể tăng hoặc giảm hạng. [Dữ liệu từng lượt](xai_results_corrected.csv) có cả `rank_after` của ảnh top-1 và `gt_rank_after` của ảnh đúng.

## Theo model seed

| dataset | model_seed | requested_k | n | final_drop | random_drop | advantage |
| --- | --- | --- | --- | --- | --- | --- |
| psg | 0 | 1 | 12 | 0.852 | 0.190 | 0.662 |
| psg | 0 | 3 | 12 | 1.604 | 0.652 | 0.952 |
| psg | 0 | 5 | 12 | 1.383 | 0.833 | 0.550 |
| psg | 1 | 1 | 12 | 0.970 | 0.144 | 0.826 |
| psg | 1 | 3 | 12 | 1.841 | 0.731 | 1.110 |
| psg | 1 | 5 | 12 | 1.983 | 1.115 | 0.868 |
| psg | 2 | 1 | 12 | 0.883 | 0.200 | 0.682 |
| psg | 2 | 3 | 12 | 1.723 | 0.796 | 0.927 |
| psg | 2 | 5 | 12 | 1.814 | 1.087 | 0.727 |
| vg | 0 | 1 | 12 | 0.481 | 0.038 | 0.443 |
| vg | 0 | 3 | 12 | 0.919 | 0.185 | 0.734 |
| vg | 0 | 5 | 12 | 1.085 | 0.240 | 0.845 |
| vg | 1 | 1 | 12 | 0.476 | 0.036 | 0.440 |
| vg | 1 | 3 | 12 | 0.882 | 0.139 | 0.743 |
| vg | 1 | 5 | 12 | 1.053 | 0.209 | 0.844 |
| vg | 2 | 1 | 12 | 0.513 | 0.077 | 0.437 |
| vg | 2 | 3 | 12 | 0.904 | 0.194 | 0.709 |
| vg | 2 | 5 | 12 | 1.050 | 0.210 | 0.840 |

Advantage vẫn dương ở cả ba seed cho mọi mức k: tại k=3, VG nằm trong khoảng **0,709–0,743**, PSG **0,927–1,110**. Sự ổn định này là ổn định trên **12 caption đã chọn cho từng dataset**, không phải phép kiểm định tổng quát trên test set. [Bảng theo seed](by_model_seed.csv) lưu các số chưa làm tròn.

## Theo nhóm truy vấn và outcome seed 0

| dataset | group | seed0_outcome | requested_k | n | final_drop | random_drop | advantage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| psg | direct | failure | 1 | 9 | 1.271 | 0.232 | 1.040 |
| psg | direct | failure | 3 | 9 | 1.788 | 0.807 | 0.981 |
| psg | direct | failure | 5 | 9 | 1.441 | 0.770 | 0.670 |
| psg | direct | success | 1 | 9 | 0.782 | 0.204 | 0.578 |
| psg | direct | success | 3 | 9 | 1.856 | 0.933 | 0.924 |
| psg | direct | success | 5 | 9 | 1.598 | 1.144 | 0.455 |
| psg | indirect | failure | 1 | 9 | 0.664 | 0.112 | 0.552 |
| psg | indirect | failure | 3 | 9 | 1.481 | 0.461 | 1.020 |
| psg | indirect | failure | 5 | 9 | 1.822 | 0.900 | 0.923 |
| psg | indirect | success | 1 | 9 | 0.888 | 0.165 | 0.723 |
| psg | indirect | success | 3 | 9 | 1.766 | 0.705 | 1.061 |
| psg | indirect | success | 5 | 9 | 2.044 | 1.233 | 0.811 |
| vg | direct | failure | 1 | 9 | 0.790 | 0.095 | 0.695 |
| vg | direct | failure | 3 | 9 | 1.164 | 0.223 | 0.941 |
| vg | direct | failure | 5 | 9 | 1.221 | 0.195 | 1.026 |
| vg | direct | success | 1 | 9 | 0.565 | 0.055 | 0.509 |
| vg | direct | success | 3 | 9 | 1.270 | 0.286 | 0.984 |
| vg | direct | success | 5 | 9 | 1.413 | 0.352 | 1.061 |
| vg | indirect | failure | 1 | 9 | 0.273 | 0.016 | 0.257 |
| vg | indirect | failure | 3 | 9 | 0.511 | 0.058 | 0.453 |
| vg | indirect | failure | 5 | 9 | 0.709 | 0.125 | 0.584 |
| vg | indirect | success | 1 | 9 | 0.334 | 0.034 | 0.299 |
| vg | indirect | success | 3 | 9 | 0.661 | 0.123 | 0.538 |
| vg | indirect | success | 5 | 9 | 0.907 | 0.207 | 0.700 |

![Fidelity advantage tại k=3 theo nhóm truy vấn direct/indirect và outcome xác định bằng seed 0](advantage_by_group.png)

*Hình 2.* Mỗi cột gồm 3 caption × 3 model seed (n=9); nhãn success/failure dựa trên seed 0. [Bảng đầy đủ cho k=1/3/5](by_group_outcome.csv).

Ở k=3, VG có advantage cao hơn trong nhóm direct (**0,941** failure; **0,984** success) so với indirect (**0,453** failure; **0,538** success). PSG không có cùng mẫu hình: hai nhóm indirect đạt **1,020** và **1,061**, trong khi direct đạt **0,981** và **0,924**. Mỗi cột chỉ có ba caption độc lập lặp qua ba seed; chênh lệch theo nhóm mang tính mô tả và chưa tách được ảnh hưởng của dataset khỏi kiến trúc GNN hay tập cạnh.

## Bootstrap 95% CI theo caption

| dataset | k | n_captions | mean_advantage | bootstrap_low | bootstrap_high |
| --- | --- | --- | --- | --- | --- |
| psg | 1 | 12 | 0.723 | 0.547 | 0.914 |
| psg | 3 | 12 | 0.996 | 0.878 | 1.104 |
| psg | 5 | 12 | 0.715 | 0.488 | 0.928 |
| vg | 1 | 12 | 0.440 | 0.288 | 0.644 |
| vg | 3 | 12 | 0.729 | 0.561 | 0.931 |
| vg | 5 | 12 | 0.843 | 0.686 | 1.014 |

CI được tạo bằng **5.000 lần bootstrap trên 12 caption** của từng dataset và k; ở mỗi caption, advantage được lấy trung bình qua ba seed trước khi lấy mẫu lại. Như vậy ba seed của cùng caption được giữ trong một cụm, tránh coi 36 dòng là 36 caption độc lập. Các khoảng ở bảng đều trên 0, phù hợp với lợi thế XAI trên **tập case đã chọn**; đây không phải CI cho toàn bộ phân phối truy vấn. [Bảng bootstrap](advantage_bootstrap_ci.csv) và [notebook tổng hợp](../../../../notebooks/summarize_vg_psg_xai_colab.ipynb) ghi lại cách tính.

## Case tiêu biểu (seed 0, k=3)

| dataset | group | seed0_outcome | caption_id | gt_rank_before | gt_rank_after | advantage | edges |
| --- | --- | --- | --- | --- | --- | --- | --- |
| psg | direct | failure | 76884 | 13 | 12 | 0.887 | motorcycle -[contains]-> image:4491; person -[contains]-> image:4491; road -[contains]-> image:4491 |
| psg | direct | success | 58392 | 1 | 1 | 1.041 | horse -[contains]-> image:11672; car -[contains]-> image:11672; road -[contains]-> image:11672 |
| psg | indirect | failure | 73603 | 13 | 12 | 1.062 | remote -[contains]-> image:14686; person -[contains]-> image:14686; tv -[contains]-> image:14686 |
| psg | indirect | success | 66698 | 1 | 4 | 0.907 | suitcase -[contains]-> image:13333; person -[contains]-> image:13333; pavement -[contains]-> image:13333 |
| vg | direct | failure | 85390 | 11 | 10 | 0.935 | zebra -[contains]-> image:18104; antelope -[contains]-> image:18104; grass -[contains]-> image:18104 |
| vg | direct | success | 6120 | 1 | 8 | 0.906 | umbrella -[contains]-> image:1223; rain -[contains]-> image:1223; people -[contains]-> image:1223 |
| vg | indirect | failure | 47795 | 20 | 20 | 0.450 | giraffe -[contains]-> image:4068; zebra -[contains]-> image:4068; fence -[contains]-> image:4068 |
| vg | indirect | success | 68975 | 1 | 1 | 0.583 | car -[contains]-> image:13786; snow -[contains]-> image:13786; meter -[contains]-> image:13786 |

Các hàng được chọn là case **gần median advantage** trong từng nhóm của seed 0 tại k=3, không phải case có tác động lớn nhất. Cột `edges` chỉ ra cạnh bị xóa; ảnh ghi sau `image:` là **ảnh top-1 ban đầu**, có thể khác ảnh đúng. [Bảng case có câu truy vấn, ID ảnh top-1 và ID ảnh đúng](representative_cases.csv); [cạnh và đường đi đầy đủ](../explanations.json).

- **VG, direct/success, caption 6120:** truy vấn “several people hold umbrellas in a rain storm”; ảnh đúng cũng là ảnh top-1 `1223`. Xóa `umbrella`, `rain`, `people → image:1223` làm ảnh đúng từ hạng **1 xuống 8**, với advantage **0,906**. Đây là ví dụ rõ về cạnh hỗ trợ dự đoán vốn đúng; phép xóa làm retrieval tệ hơn.
- **PSG, indirect/success, caption 66698:** truy vấn “A man walking down the street carrying a suitcase.”; ảnh đúng/top-1 là `13333`. Xóa ba cạnh `suitcase`, `person`, `pavement → image:13333` làm ảnh đúng từ hạng **1 xuống 4**, advantage **0,907**. Fidelity cao không đồng nghĩa perturbation giúp ảnh đúng.
- **PSG, direct/failure, caption 76884:** ảnh top-1 ban đầu là `4491`, ảnh đúng `15368` ở hạng **13**. Xóa `motorcycle`, `person`, `road → image:4491` làm ảnh top-1 cũ tụt xuống hạng **46** và ảnh đúng lên hạng **12**. Advantage **0,887** cho thấy các cạnh này hỗ trợ một dự đoán sai, nhưng cải thiện ảnh đúng chỉ một bậc.

### Mười case có câu truy vấn và ảnh gốc

Tập XAI hiện tại có **12 caption/bộ: 6 success và 6 failure**, mỗi loại gồm 3 direct và 3 indirect theo hạng checkpoint seed 0. Bảng dưới chọn **10 trong 24 caption đã chạy**, gồm **5 success, 5 failure; 5 VG, 5 PSG; 5 direct, 5 indirect**. Tám case ở bảng trên được chọn gần median advantage của từng nhóm; thêm VG 60916 (success) và PSG 46151 (failure) để đủ 5/5. Đây là bộ minh họa, không phải một lượt thí nghiệm mới hay tập dùng để tính các trung bình phía trên. [Danh mục 10 case và URL ảnh gốc](case_studies_10.csv); [danh sách 24 case của thí nghiệm](../selected_cases.json).

Hình được lưu ngay trong `case_images/` để mở báo cáo khi không có mạng. Tất cả kết quả hạng và advantage dưới đây dùng **model seed 0, k=3**. `Success` nghĩa là ảnh đúng ban đầu ở hạng 1; `failure` nghĩa là ảnh đúng ban đầu ở hạng trên 10. Hạng sau perturbation có thể tốt hơn, xấu hơn hoặc giữ nguyên.

**5 success:** ảnh đúng cũng chính là ảnh top-1 mà XAI giải thích.

| Dataset · caption · nhóm | Câu truy vấn | Ảnh đúng = top-1 trước xóa cạnh | Hạng ảnh đúng trước → sau | Advantage |
|---|---|---|---:|---:|
| VG 6120 · direct | several people hold umbrellas in a rain storm | <img src="case_images/vg_6120_gt.jpg" alt="Ảnh đúng VG 6120" width="145"><br>ảnh 1223 | 1 → 8 | 0.906 |
| VG 60916 · direct | A black and white photo of a building with a clock on it. | <img src="case_images/vg_60916_gt.jpg" alt="Ảnh đúng VG 60916" width="145"><br>ảnh 12175 | 1 → 4 | 0.711 |
| VG 68975 · indirect | Car buried in snow next to what might be a gas pump. | <img src="case_images/vg_68975_gt.jpg" alt="Ảnh đúng VG 68975" width="145"><br>ảnh 13786 | 1 → 1 | 0.583 |
| PSG 58392 · direct | A horse walks between cars in a parking lot. | <img src="case_images/psg_58392_gt.jpg" alt="Ảnh đúng PSG 58392" width="145"><br>ảnh 11672 | 1 → 1 | 1.041 |
| PSG 66698 · indirect | A man walking down the street carrying a suitcase. | <img src="case_images/psg_66698_gt.jpg" alt="Ảnh đúng PSG 66698" width="145"><br>ảnh 13333 | 1 → 4 | 0.907 |

**5 failure:** XAI giải thích ảnh top-1 **sai**; bảng hiển thị cả ảnh đúng và ảnh top-1 để so sánh. Cột “hạng top-1 cũ” theo dõi **chính ảnh từng đứng đầu**, không phải ảnh đứng đầu mới sau perturbation.

| Dataset · caption · nhóm | Câu truy vấn | Ảnh đúng | Ảnh top-1 cũ | Hạng ảnh đúng trước → sau | Hạng top-1 cũ trước → sau | Advantage |
|---|---|---|---|---:|---:|---:|
| VG 85390 · direct | Two zebras and another animal grazing in the grass. | <img src="case_images/vg_85390_gt.jpg" alt="Ảnh đúng VG 85390" width="145"><br>ảnh 17067 | <img src="case_images/vg_85390_top1.jpg" alt="Ảnh top-1 sai VG 85390" width="145"><br>ảnh 18104 | 11 → 10 | 1 → 15 | 0.935 |
| VG 47795 · indirect | a giraffe in an enclosure at an animal park. | <img src="case_images/vg_47795_gt.jpg" alt="Ảnh đúng VG 47795" width="145"><br>ảnh 9552 | <img src="case_images/vg_47795_top1.jpg" alt="Ảnh top-1 sai VG 47795" width="145"><br>ảnh 4068 | 20 → 20 | 1 → 7 | 0.450 |
| PSG 76884 · direct | A person sits on top of a motorcycle with others. | <img src="case_images/psg_76884_gt.jpg" alt="Ảnh đúng PSG 76884" width="145"><br>ảnh 15368 | <img src="case_images/psg_76884_top1.jpg" alt="Ảnh top-1 sai PSG 76884" width="145"><br>ảnh 4491 | 13 → 12 | 1 → 46 | 0.887 |
| PSG 73603 · indirect | A group of women playing video games together | <img src="case_images/psg_73603_gt.jpg" alt="Ảnh đúng PSG 73603" width="145"><br>ảnh 14713 | <img src="case_images/psg_73603_top1.jpg" alt="Ảnh top-1 sai PSG 73603" width="145"><br>ảnh 14686 | 13 → 12 | 1 → 13 | 1.062 |
| PSG 46151 · indirect | There are many people on this field flying kites | <img src="case_images/psg_46151_gt.jpg" alt="Ảnh đúng PSG 46151" width="145"><br>ảnh 9224 | <img src="case_images/psg_46151_top1.jpg" alt="Ảnh top-1 sai PSG 46151" width="145"><br>ảnh 6080 | 20 → 20 | 1 → 18 | 0.733 |

Hai bảng cũng cho thấy mức giảm điểm fuse của ảnh top-1 và thay đổi hạng ảnh đúng là hai hiện tượng khác nhau: VG 68975 và PSG 58392 có advantage dương nhưng ảnh đúng vẫn hạng 1; VG 47795 và PSG 46151 làm ảnh top-1 sai tụt hạng nhưng ảnh đúng không nhích lên. Tất cả ảnh được đối chiếu theo `ground_truth_image_id` và `image_id` trong [CSV kết quả XAI](xai_results_corrected.csv), rồi lấy URL từ metadata của từng bundle.

### Loại cạnh và đường đi được giải thích

Đếm trực tiếp từ [216 explanation](../explanations.json): tại k=1, **72/72** cạnh được chọn của hai bộ đều là `contains`. Tại k=3, VG có **102/108** cạnh `contains` và 6 `concept`; PSG có **104/108** `contains` và 4 `sg_rel`. Tại k=5, VG có **151/180** `contains` và 29 `concept`; PSG có **169/180** `contains`, 10 `sg_rel` và 1 `concept`. Vì vậy, kết quả fidelity hiện chủ yếu phản ánh cạnh nối nhãn với ảnh top-1; bằng chứng về giải thích quan hệ nhiều bước còn ít. Tại k=3, chỉ có **6 đường đi dài 2 cạnh** ở VG và **4** ở PSG; các đường đi còn lại dài 1 cạnh. Một số cạnh được chọn không nằm trên đường nối đến ảnh top-1, nên số `n_explained_paths` có thể nhỏ hơn k.

Trong [CSV đã sửa seed](xai_results_corrected.csv), `structural_validity` và `path_validity` đều bằng **1,0** trên 216 dòng, tức cạnh và đường đi được xuất ra đúng schema/cấu trúc graph. `evidence_known_rate` và `evidence_support_rate` đều bằng **0,0**: metadata hiện để trạng thái hỗ trợ là `uncertain`, chưa có xác minh độc lập từ ảnh hay annotation. Do đó, không thể dùng hai chỉ số validity cấu trúc để khẳng định cạnh là bằng chứng ngữ nghĩa đúng. Ở k=3, sparsity trung bình là **94,684%** (VG) và **93,655%** (PSG) *trong vùng đã sàng lọc*; chính vùng sàng lọc chỉ bao phủ trung bình **0,869%** và **0,477%** số cạnh 2-hop tương ứng.

## Tính toàn vẹn và tái lập

Trong [CSV thô](../xai_results.csv), cột `seed` vô tình ghi **random seed** của evaluator; sáu JSON kết quả theo seed ở thư mục cha vì thế rỗng. Notebook tổng hợp đã đối chiếu 216 dòng với `summary.json` và 216 explanation, khôi phục `model_seed` từ quy tắc tạo random seed, rồi lưu [CSV đã sửa](xai_results_corrected.csv) và sáu JSON `*_seed*_corrected.json` mà không sửa dữ liệu thô. Các bảng và hình trong báo cáo này dùng bản đã sửa. Notebook XAI hiện đã sửa cách ghi seed cho lượt chạy sau.

Báo cáo được bổ sung diễn giải thủ công trên artifact của lượt này. Nếu chạy lại [notebook tổng hợp](../../../../notebooks/summarize_vg_psg_xai_colab.ipynb), cell cuối sẽ tạo lại `report.md` dạng bảng ngắn và ghi đè phần diễn giải; hãy lưu bản báo cáo này trước khi chạy lại cell đó.

## Giới hạn

- Chọn 12 caption mỗi dataset theo outcome seed 0; không suy rộng ra toàn bộ test.
- Một caption xuất hiện ở ba seed; CI lấy mẫu theo caption để giữ cụm này.
- VG và PSG có dữ liệu, GNN và tập cạnh khác nhau; không quy chênh lệch chỉ cho dataset.
- Coverage là tỷ lệ cạnh vùng 2-hop được xét; structural validity không xác nhận hỗ trợ thị giác.
- Top-1 đổi hạng không nhất thiết làm ảnh đúng tăng hạng.
- CSV gốc được giữ nguyên; `xai_results_corrected.csv` tách model seed và random seed.
