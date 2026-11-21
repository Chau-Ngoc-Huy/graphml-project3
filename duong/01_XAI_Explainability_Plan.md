# Kế hoạch XAI cho MMKG text-to-image retrieval — giai đoạn đường quan hệ

## 1. Mục tiêu và kết quả hiện có

Mục tiêu là giải thích **vì sao một ảnh được scorer xếp hạng cao** bằng cạnh/đường đi thực sự ảnh hưởng đến điểm, đồng thời kiểm tra mạch giải thích có phù hợp với truy vấn và ảnh hay không. Báo riêng **fidelity đối với mô hình**, **tính logic/căn cứ thị giác**, **độ cô đọng**, **độ bao phủ** và **hạng truy xuất**; không gộp chúng thành một điểm “validity”. Cách tách các chiều đánh giá này phù hợp với [BAGEL](https://arxiv.org/abs/2206.13983); kiểm tra tính đúng của từng bước suy luận tham khảo [ReCEval](https://aclanthology.org/2023.emnlp-main.622/).

[Lượt VG/PSG đã chạy](runs/qa_vg_psg_xai/xai-20261121-130842/analysis/report.md) là **baseline cố định**, không thay số liệu sau khi thiết kế phép thử mới: 24 caption (12/bộ), 3 model seed, k=1/3/5, 216 dòng; mỗi bộ có 6 success và 6 failure theo seed 0. Xóa cạnh XAI làm điểm top-1 giảm hơn baseline ngẫu nhiên, nhưng ở k=3 có **102/108 cạnh VG** và **104/108 cạnh PSG** là `contains`; structural/path validity bằng 1, còn evidence support/known bằng 0. Vùng cạnh được sàng lọc chỉ chiếm trung bình dưới 1% vùng 2-hop. Vì vậy lượt này chứng minh **đóng góp của cạnh đối tượng đối với dự đoán**, chưa chứng minh suy luận quan hệ hay validity ngữ nghĩa.

**Nguyên tắc:** đường dài hơn không tự làm explanation đúng hơn. Đặt tiêu chí riêng cho *giải thích quan hệ*: đường nối khái niệm của truy vấn với ảnh qua ít nhất **một cạnh không phải `contains`**, rồi kết thúc bằng cạnh `contains`. Với scorer hai tầng hiện tại, ưu tiên đường 2–3 cạnh, tương ứng 1–2 bước trên graph nhãn trước khi tới ảnh. Đường 1 cạnh `contains` vẫn được báo là *bằng chứng đối tượng* cho truy vấn direct, nhưng không tính là mạch suy luận quan hệ. Nếu không có đường quan hệ hợp lệ và có tác động, ghi rõ **không tìm thấy**, không thêm cạnh chỉ để đạt độ dài.

## 2. Hai câu hỏi phải được đánh giá riêng

1. **Mô hình có dùng đường này không?** Xóa cạnh/đường đi, chạy lại QA scorer và phép fuse trên cùng 100 ứng viên; đo đổi điểm/hạng so với can thiệp ngẫu nhiên khớp ngân sách. Đây là *faithfulness*.
2. **Đường này có là lý do hợp lý cho cặp truy vấn–ảnh không?** Từng bước phải có ý nghĩa suy luận, đúng chiều hoặc có luật đảo chiều được phép, và được ảnh/annotation hỗ trợ. Đây là *logical/visual validity*. Đường trong graph và giảm score chưa đủ trả lời câu hỏi này.

Luôn ghi `target_image_id` và vai trò của nó: **top-1 dự đoán**, **ảnh đúng**, hay **ảnh âm khó**. Với case failure, xóa cạnh hỗ trợ top-1 sai có thể làm ảnh đúng tăng hạng; không gọi tỷ lệ `gt_improved` là chất lượng XAI. Ghi `gt_rank_before/after` và `target_rank_before/after` riêng.

## 3. Điều kiện để mô hình và graph cho phép giải thích quan hệ

- [ ] Ghi rõ graph thực sự đi vào forward pass, số tầng và hướng truyền tin. Chỉ nhận đường đi trong vùng ảnh hưởng của scorer. VG hiện dùng SAGE + ConceptNet, không có `sg_rel`; PSG dùng GAT + scene graph + ConceptNet. Báo hai cấu hình riêng, không quy chênh lệch chỉ cho dataset.
- [ ] Kiểm tra giới hạn hiện tại: [`QAScorer.label_h`](../huy/src/mmxai/graph/qa.py) nhận `edge_index` của graph nhãn, **không nhận tên predicate làm đặc trưng cạnh**. [`xai/model.py`](xai/model.py) giữ `pair_relation_hint` để hiển thị, nhưng hint không chứng minh mô hình đã dùng ý nghĩa `on`, `IsA`, v.v. Với checkpoint hiện có, chỉ kết luận về **kết nối/lan truyền trên cạnh**. Muốn tuyên bố mô hình hiểu quan hệ cụ thể, cần scorer có biểu diễn relation-aware, checkpoint tương ứng và ablation giữ nguyên đối tượng nhưng đổi quan hệ.
- [ ] Với từng cạnh, lưu ID ổn định, source/target, loại (`contains`/`sg_rel`/`concept`), predicate gốc, chiều suy luận được phép, nguồn (`train scene graph`, ConceptNet, nhãn ảnh), split và trạng thái bằng chứng. Không dùng quan hệ tổng hợp từ ảnh test làm tri thức huấn luyện.
- [ ] Phân biệt cạnh `contains` của ảnh đích với cạnh quan hệ giữa nhãn. Một đường hợp lệ về cấu trúc có dạng khái niệm truy vấn → cạnh quan hệ → nhãn ảnh → `contains` → ảnh. Việc duyệt ngược một cạnh chỉ hợp lệ nếu predicate và luật suy luận cho phép; không coi mọi cặp node kề nhau là phép suy luận hai chiều.

**Cổng quyết định:** nếu chỉ có checkpoint không nhận predicate, tiếp tục đánh giá *đường ảnh hưởng có cạnh quan hệ* nhưng ghi nhãn là **relational-connectivity explanation**. Đánh giá *predicate-level logical reasoning* là mục tiêu của biến thể relation-aware, không suy từ checkpoint hiện tại.

## 4. Bộ truy vấn và ảnh đánh giá

- [ ] Giữ 24 caption hiện tại để so sánh baseline trên cùng protocol. Bổ sung bộ đánh giá độc lập, tối thiểu **10 truy vấn direct và 10 truy vấn relational/indirect cho mỗi dataset được báo cáo**. Tạo câu có đối tượng và quan hệ rõ ràng; tránh câu chung chung hoặc chỉ liệt kê danh từ. Khóa danh sách trước khi xem XAI của lượt mới.
- [ ] Với mỗi truy vấn, có ảnh dương và **ảnh âm khó chứa cùng các đối tượng nhưng khác quan hệ/vai trò**; kiểm tra ảnh bằng mắt và lưu `relevance`/`uncertain` cùng căn cứ. Ví dụ truy vấn “người cầm ô” cần phân biệt ảnh người cầm ô với ảnh chỉ có người và ô. Đây là phép thử về composition, không phải câu chữ ngẫu nhiên; [ARO](https://arxiv.org/abs/2210.01936) là ví dụ về đánh giá quan hệ/thứ tự bằng tương phản có kiểm soát.
- [ ] Có thể dùng [24 truy vấn của Khánh](../khanh/queries/README.md) làm **ứng viên**. Không lấy positive 2-hop chưa kiểm tra bằng mắt làm ground truth. [Thực nghiệm VG của Khánh](../khanh/results/level2_vg_own_experiment.md) cho thấy baseline chỉ đếm đối tượng đạt MRR cao hơn mô hình quan hệ trên bộ đó; do vậy cần ảnh âm cùng đối tượng và kiểm chứng độc lập.
- [ ] Khai báo rõ `direct/indirect` của bộ **mới** theo mức suy luận mà ảnh và truy vấn đòi hỏi, không đồng nhất với nhãn tự động của Thắng (`indirect` = thiếu ít nhất một nhãn đối tượng). Lưu cả nhãn nguồn và nhãn đã xác minh.
- [ ] Giữ ít nhất **5 case success và 5 case failure** theo hạng của chính checkpoint trước perturbation, cân bằng direct/indirect nếu đủ dữ liệu. Các case không có đường quan hệ vẫn ở mẫu số để đo độ bao phủ. Case ảnh đúng ngoài top-100 được báo riêng vì reranker top-100 không thể đổi hạng của ảnh đó.
- [ ] Truy vấn mới cần CLIP text embedding **cùng encoder** với bundle và bộ khái niệm liên kết tương thích QA scorer; không thể thay chuỗi caption trong notebook mà giữ embedding/`cap_concepts` cũ. Giữ split/candidate pool/α/seed cố định; tuning trên validation, không dùng tập kiểm tra cuối.

## 5. Trích xuất đường giải thích

### 5.1. Xác định tập cạnh đủ điều kiện trước khi chấm XAI

- [ ] Với cặp `(query, image)`, lấy tất cả cạnh `contains` của ảnh, cộng các cạnh `concept`/`sg_rel` nằm trên đường từ khái niệm truy vấn tới nhãn ảnh trong **receptive field thật** của mô hình. Lưu `E_eligible` và `E_eligible_relation` trước bước xếp hạng explanation.
- [ ] Nếu phải giới hạn tính toán, sàng lọc quan hệ bằng quy tắc tất định chỉ dùng query, graph và checkpoint (không dùng kết quả perturbation hoặc nhãn test). Lưu `E_screened`, ngân sách và coverage **theo từng loại cạnh**. Không dùng duy nhất `E_screened` nhỏ làm mẫu số sparsity; kiểm tra độ nhạy khi tăng ngân sách thay vì cố định top-40.
- [ ] Tạo các đường đơn từ khái niệm truy vấn đến ảnh: 1 cạnh `contains` cho object-level; 2–3 cạnh với ≥1 `concept`/`sg_rel` cho relational-level. Lưu cạnh, predicate/chiều, nguồn, độ dài, khái niệm truy vấn được nối và nhãn ảnh kết thúc. Cạnh lẻ không nối được chỉ được gọi là **top edge**, không gán thành path.

### 5.2. Xếp hạng theo đóng góp của cả đường

- [ ] Tính leave-one-edge-out để đối chiếu lượt cũ, nhưng **đo path contribution bằng xóa đồng thời toàn bộ cạnh của path** rồi chạy lại scorer. Tách `graph_score_drop`, `QA_score_drop` và `final_score_drop`; không cộng đóng góp cạnh đơn như thể scorer tuyến tính.
- [ ] So sánh hai chế độ có cùng ngân sách: `contains`-only và đường có cạnh quan hệ. Chọn đường có tác động dương **trong chế độ relational**; nếu không có, trả về `no_relational_explanation`. Không ép thêm cạnh quan hệ yếu/âm để làm đường dài.
- [ ] Với nhiều path chồng lấn, chọn theo mức tăng đóng góp khi thêm path vào tập đã chọn, đếm cạnh trùng một lần. Báo cả số path và số cạnh duy nhất; ngân sách so sánh theo **số cạnh duy nhất**, không chỉ số path.
- [ ] Không xóa cạnh ngoài graph được mô hình sử dụng. Chạy lại `label_h` khi bỏ cạnh nhãn–nhãn, cập nhật nhãn ảnh khi bỏ `contains`, rồi tính lại QA/fuse trên đúng tập ứng viên.

## 6. Metric và protocol đánh giá

| Trục | Cách đo và điều kiện diễn giải |
|---|---|
| **Fidelity** | Mức giảm `graph_score`/QA/fuse và đổi hạng của ảnh đích khi bỏ explanation; so với random **path** cùng số cạnh, độ dài, loại cạnh và vùng đủ điều kiện. Báo thêm ablation `contains`-only, `concept`-only, `sg_rel`-only, xáo cạnh quan hệ và giữ cạnh quan hệ nhưng đổi predicate nếu model relation-aware. |
| **Relational yield/coverage** | Tỷ lệ **mọi case đã đăng ký trước** có ít nhất một path dài 2–3 cạnh, chứa ≥1 cạnh không phải `contains`, đúng receptive field và có đóng góp dương. Báo số case không có đường, không loại khỏi mẫu số. |
| **Sparsity** | `1 − |E_exp_unique| / |E_eligible|`; báo thêm `|E_exp_relation| / |E_eligible_relation|`, `|E_exp_unique|`, số path, độ dài, overlap và `|E_screened| / |E_eligible|`. Nếu mẫu số relation bằng 0, ghi `N/A` và tính case đó vào nhóm không đủ điều kiện, không tự cho điểm sparsity tối đa. So sánh fidelity–sparsity theo **cùng ngân sách cạnh**; không dùng sparsity cao làm kết luận nếu coverage/validity thấp. |
| **Structural/path validity** | Cạnh tồn tại, node type đúng, đường liên tục và kết thúc ở ảnh; đây là kiểm tra dữ liệu/phần mềm. Báo nhưng không coi gần 1,0 là chứng cứ logic. |
| **Logical validity** | Trên path có thật, kiểm tra từng bước: nguồn/đích liên quan đến claim của truy vấn, predicate và chiều/luật suy luận hợp lệ, bước mới bổ sung thông tin, toàn chuỗi hỗ trợ claim. `contains` chỉ chứng minh đối tượng được gán cho ảnh, không tự chứng minh hành động/vị trí giữa các đối tượng. |
| **Visual evidence** | Đối chiếu từng đối tượng/quan hệ với ảnh hoặc annotation độc lập; không dùng lại chính nhãn đối tượng đã cấp cho scorer làm “gold” xác minh. Gán `supported`/`contradicted`/`uncertain`, `support_source` và ghi lý do. Báo tỷ lệ known và supported trên **toàn bộ cạnh/path được chọn**, cùng tỷ lệ `no_path`. |

Với **ít nhất 10 case study có ảnh**, hai người chấm độc lập theo rubric (đối tượng, quan hệ, chiều suy luận, toàn chuỗi, căn cứ ảnh), giải quyết bất đồng và báo mức đồng thuận. Đánh giá máy/LLM có thể hỗ trợ sàng lọc, nhưng không thay thế nhãn kiểm chứng gốc. Path chỉ được gọi là *logically supported* khi mọi bước cần thiết được hỗ trợ và không có mâu thuẫn; thiếu bằng chứng là `uncertain`, không mặc định đúng.

**Thống kê:** cùng query xuất hiện ở ba seed nên bootstrap theo **query/cặp truy vấn đối chứng**, giữ seed trong cụm; báo CI 95%, số case và cả phân bố, không coi 36 dòng là 36 truy vấn độc lập. Không so sánh trực tiếp giá trị fuse giữa VG và PSG nếu thang điểm/cấu hình khác. Giữ cố định CLIP, ứng viên, checkpoint, α và query embedding trong từng so sánh can thiệp.

## 7. Ma trận thí nghiệm và thứ tự thực hiện

| Giai đoạn | Thực hiện | Điều kiện qua giai đoạn |
|---|---|---|
| **A. Audit baseline** | Khóa lượt 24 caption đã commit; xác nhận loại cạnh/path, coverage và mốc score. | Báo cáo baseline tái lập được, không viết lại kết quả cũ. |
| **B. Query/evidence** | Tạo truy vấn đối chứng và ảnh dương/âm; kiểm tra trực quan, lưu provenance. | Có ≥10 direct + ≥10 relational cho mỗi bộ được báo cáo, relevance đã kiểm chứng, ≥5 success + ≥5 failure để phân tích. |
| **C. Path search** | Mở rộng vùng đủ điều kiện theo receptive field, dựng path 2–3 cạnh, đo đóng góp path và yield. | Có số đo coverage theo loại cạnh; không che giấu `no_relational_explanation`. |
| **D. Perturbation** | Xóa path thật vs random path cùng ngân sách; ablation theo loại cạnh, k và seed. | Phép can thiệp thật sự chạy lại model; mọi so sánh dùng cùng candidate pool. |
| **E. Validity** | Chấm logic và ảnh cho case study; tách structural, semantic, visual và unknown. | Có rubric/annotation độc lập, các claim trong báo cáo phù hợp với bằng chứng. |
| **F. Relation-aware (nếu cần)** | Nếu checkpoint hiện tại chỉ cho connectivity hoặc không tạo path có tác động, huấn luyện/đánh giá biến thể dùng predicate; so với `contains`-only trên cùng truy vấn. | Chỉ tuyên bố suy luận predicate khi scorer thực sự nhận predicate và vượt baseline đối chứng. |

Nếu giai đoạn C không tìm được đường quan hệ có tác động, kết luận thực nghiệm là **mô hình/thiết kế hiện tại chưa cung cấp giải thích quan hệ**. Không tăng độ dài bằng cách ghép cạnh hậu kiểm hoặc chỉ giữ các case “đẹp”.

## 8. Đầu ra và tiêu chí nghiệm thu

- `queries_relational.*`: câu truy vấn, claim `(subject, predicate, object)`, nhóm, ảnh dương/âm, relevance đã kiểm chứng, split, CLIP embedding/concepts và provenance.
- `paths.jsonl`: toàn bộ candidate path và lý do được/không được chọn; gồm node, edge, predicate, chiều, nguồn, độ dài, loại, contribution; `no_path` là bản ghi rõ ràng.
- `xai_results.csv`: một dòng cho `(dataset, query, target_image, model_seed, budget, intervention)` với điểm/rank trước–sau, denominator eligible/screened, edge-type counts và random baseline tương ứng.
- `validity_annotations.csv`: chấm từng bước, bằng chứng ảnh, `supported`/`contradicted`/`uncertain`, người chấm và quyết định cuối.
- Notebook Colab và báo cáo: hình fidelity–sparsity–coverage, tỷ lệ explanation có cạnh quan hệ, CI theo query, 10 case có ảnh/câu truy vấn/GT/top-1/cạnh và kết luận riêng VG/PSG. Không ghi đè snapshot baseline.

**Nghiệm thu:** (1) ít nhất 20 truy vấn đã xác minh cho mỗi bộ được báo cáo; (2) ít nhất 10 case study, gồm 5 success và 5 failure; (3) mọi *relational explanation* được xuất có path thực sự dài 2–3 cạnh, ≥1 cạnh không phải `contains`, nối truy vấn tới ảnh trong vùng scorer dùng; (4) báo cả tỷ lệ không tìm được path; (5) fidelity vượt baseline ngẫu nhiên có cùng ngân sách trên tập truy vấn định trước; (6) sparsity tính trên tập cạnh đủ điều kiện và kèm coverage; (7) validity logic/thị giác được chấm độc lập, không suy từ schema; (8) mọi kết luận về predicate có kiểm tra relation-aware và `contains`-only control. Không đặt trước ngưỡng yield/fidelity để chọn case thuận lợi sau khi xem kết quả.
