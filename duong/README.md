# Nhiệm vụ 3 — Explainability (xAI) cho Text-to-Image Retrieval

> **Bài toán:** Knowledge Graph–Enhanced Text-to-Image Retrieval
> **Mục đích tài liệu:** Mô tả phạm vi nhiệm vụ, kiến trúc tích hợp, tổ chức mã nguồn, đầu vào/đầu ra và tiêu chí nghiệm thu.
> **Kế hoạch chi tiết:** [`01_XAI_Explainability_Plan.md`](./01_XAI_Explainability_Plan.md).

## 1. Tổng quan nhiệm vụ

Mục tiêu là giải thích vì sao một ảnh được xếp hạng cao bằng **cạnh, đường đi hoặc đồ thị con thực sự ảnh hưởng đến graph score/final score**, sau đó đánh giá định lượng **fidelity, sparsity và validity**.

**Nguyên tắc xuyên suốt:** Không tạo lời giải thích chỉ vì đường đi nghe có vẻ hợp lý. Explanation phải xuất phát từ đúng đồ thị, scorer và cơ chế xếp hạng đang sử dụng; có thể kiểm chứng bằng can thiệp lên cạnh/đường đi và chạy lại mô hình.

### 1.1. Phạm vi tối thiểu

| Thành phần | Phạm vi cần thực hiện | Đầu ra chính |
|---|---|---|
| xAI extractor | Trích xuất các cạnh/đường đi có đóng góp cho từng `(query, image)` | Edge/path list, contribution, graph con |
| xAI evaluation | Fidelity, sparsity, structural validity, evidence support; baseline xóa cạnh ngẫu nhiên | Bảng metric và phân tích trường hợp |
| Truy vấn đánh giá | Tối thiểu 10 truy vấn phân tích định lượng; ưu tiên dùng bộ ≥20 truy vấn trực tiếp/gián tiếp của nhóm | `queries.csv`, case studies |
| Thí nghiệm Colab | Chạy checkpoint VG/PSG → giải thích → kiểm tra thay đổi score/rank | Notebook và CSV/JSON chạy lại được |

## 2. Kiến trúc tổng thể

```mermaid
flowchart TD
    Q[Truy vấn văn bản] --> C[CLIP / OpenCLIP: chọn top-k ảnh]
    C --> R[Retrieval scorer: CLIP + Graph]
    G[Scene graph / retrieval graph] --> R
    R --> O[Top-k ảnh + điểm số]
    R --> X[xAI: trích xuất cạnh / đường đi đóng góp]
    G --> X
    X --> M[Đánh giá Fidelity / Sparsity / Validity]
    R --> M
    O --> D[Notebook XAI trên Colab]
    X --> D
```

Module xAI nhận graph, danh sách ảnh ứng viên và scorer; tạo explanation, can thiệp lên graph rồi đo ảnh hưởng đến score/rank. Điểm tích hợp với pipeline của nhóm là `image_id`, `query_id`, graph schema, scorer interface, split/seed và các bảng kết quả được thống nhất.

## 3. Cấu trúc hiện dùng

```text
duong/
├── xai/
│   ├── model.py            # QA scorer thật cho VG/PSG và graph can thiệp
│   ├── graph.py            # Node, Edge, schema, local graph
│   ├── scorer_adapter.py   # Giao diện score trên tập ứng viên cố định
│   ├── perturbation.py     # Xóa cạnh, chấm điểm lại, random baseline, rank
│   ├── explain.py          # Leave-one-edge-out và top-k cạnh
│   ├── evaluate_xai.py     # Fidelity, sparsity, validity
│   ├── io.py               # Xuất explanation
│   └── queries.csv         # Catalog truy vấn tham khảo, chưa dùng ở hai notebook
├── notebooks/
│   ├── train_vg_psg_for_xai_colab.ipynb
│   └── xai_vg_psg_colab.ipynb
└── tests/                  # Fixture nhỏ và test cho lõi XAI
```

Checkpoint, bundle và kết quả lớn nằm trên MyDrive; repo chỉ giữ mã và notebook.

## 4. Các module cần thực hiện

### 4.1. Module xAI

- **`scorer_adapter.py`:** Thống nhất API tính `clip_score`, `graph_score`, `final_score` và kết nối scorer GNN thật.
- **`explain.py`:** Xác định đồ thị con ảnh hưởng tới dự đoán, đo đóng góp từng cạnh/đường đi và chọn explanation ngắn gọn.
- **`perturbation.py`:** Thực hiện xóa/mask cạnh explanation hoặc cạnh ngẫu nhiên; **tính lại graph-dependent score** trên graph sau can thiệp.
- **`evaluate_xai.py`:** Đo giảm score, đổi hạng, sparsity, validity và so sánh với random baseline.

Các metric cốt lõi:

| Metric | Ý nghĩa | Phép kiểm tra |
|---|---|---|
| Fidelity | Cạnh giải thích có ảnh hưởng thật đến dự đoán? | Xóa cạnh explanation, tính lại score/rank; đối chiếu xóa cạnh ngẫu nhiên cùng số lượng |
| Sparsity | Giải thích có cô đọng? | `1 - #explained_edges / #local_candidate_edges` |
| Structural validity | Quan hệ/đường đi có hợp lệ trong graph? | Kiểm tra edge IDs, node types, relation schema và tính liên thông |
| Evidence support | Giải thích có căn cứ từ dữ liệu nguồn? | Kiểm tra metadata/caption/triple provenance; rà soát mẫu |

Fidelity cần báo riêng biến động **graph score** và **final score**. Các so sánh thứ hạng phải sử dụng cùng candidate pool, cùng mô hình và cùng tham số.

## 5. Tích hợp với pipeline của nhóm

Các thành phần bên ngoài nhiệm vụ 3 được coi là **dependencies**, không phải phần cần triển khai lại:

| Đầu vào cần nhận | Nguồn từ nhóm | Yêu cầu tích hợp |
|---|---|---|
| CLIP features/scores, `candidate_image_ids`, split | Nhánh baseline retrieval | IDs cố định; score truy cập được cho mọi ứng viên |
| Graph schema, graph builder, GNN checkpoint/forward | Nhánh graph-enhanced retrieval | Có thể chạy lại forward khi thay đổi cạnh |
| Reranker tốt nhất, bộ truy vấn direct/indirect | Nhánh phương pháp mới | Cùng input/output và cùng evaluation protocol |

Đề xuất interface logic:

```python
# Pseudocode/API contract — các hàm sẽ được triển khai trong repo.
scores = scorer.score(query_text, candidate_image_ids, graph)
# scores[image_id] = {"clip_score": ..., "graph_score": ..., "final_score": ...}

explanation = explainer.explain(query_text, image_id, graph, scorer)
# explanation = {"edge_ids": [...], "paths": [...], "contributions": {...}}

metrics = evaluator.evaluate(
    query_text, image_id, candidate_image_ids, graph,
    scorer, explanation
)
# metrics = {"fidelity": ..., "sparsity": ..., "structural_validity": ...,
#            "evidence_support": ..., "rank_drop": ...}
```

Khi thành phần upstream chưa sẵn sàng, dùng graph/scorer mẫu **chỉ để phát triển và kiểm thử**. Thí nghiệm cuối phải dùng đúng scorer/model retrieval thực tế của nhóm.

## 6. Quy trình thực thi ở mức module

1. **Chuẩn hóa cấu hình và schema:** Thống nhất ID, node/relation types, score interface, dữ liệu/split và cơ chế lưu provenance.
2. **Xây xAI độc lập:** Dùng scorer mẫu để kiểm thử contribution, edge removal, random deletion, metric và biểu đồ.
3. **Tích hợp retrieval:** Kết nối adapter tới CLIP/GNN/reranker của nhóm, bảo đảm xóa cạnh dẫn tới chạy lại graph scoring.
4. **Đánh giá xAI:** Chạy trên tập truy vấn cố định, so sánh explanation vs random và phân tích 5 trường hợp thành công/5 thất bại.
5. **Đóng gói demo và tài liệu:** Xuất CSV, hình biểu đồ, notebook, hướng dẫn chạy và hạn chế của kết quả.

## 7. Tiêu chí nghiệm thu

- [ ] Có xAI extractor gắn với **cơ chế score thật**, không chỉ tạo đường đi hậu kiểm.
- [ ] Có fidelity, sparsity, validity và phép thử explanation-vs-random trên **ít nhất 10 truy vấn**.
- [ ] Có phân tích **5 retrieval success + 5 retrieval failure** và notebook xuất explanation có provenance.
- [ ] Có split/seed cố định, không dùng test để tuning hoặc tạo cạnh thống kê, và các thí nghiệm chính có thể chạy lại.
- [ ] Có hướng dẫn cài dependencies và các entry point chạy; không cần dựng web app hoàn chỉnh.

## 8. Ranh giới phạm vi và nguyên tắc thực hiện

- Không huấn luyện lại encoder thị giác lớn từ đầu; tận dụng mô hình của nhóm hoặc pretrained encoder.
- Không mở rộng nhiều phương pháp explainer/graph database nếu chưa hoàn thành pipeline định lượng tối thiểu.
- Không lấy nhãn relevance hoặc chọn hyperparameter từ chính tập test.
- Không nhầm **tính hợp lệ của cạnh** với **mức ảnh hưởng thực tế của cạnh lên quyết định xếp hạng**.
- Ưu tiên mã có thể kiểm thử, kết quả tái lập, nguồn dữ liệu minh bạch và demo ngắn gọn.

## 9. Lõi XAI hiện có

Các module `graph.py`, `scorer_adapter.py`, `perturbation.py`, `explain.py`, `evaluate_xai.py` và `io.py` được notebook XAI gọi trực tiếp hoặc gián tiếp. `model.py` chuyển graph can thiệp thành phép xóa cạnh thật trong QA scorer của Huy. Xóa cạnh quan hệ làm chạy lại `label_h`; xóa cạnh `contains` cập nhật nhãn ảnh, sau đó `graph_matrix` và phép fuse được tính lại trên đủ top-100 ứng viên. Cạnh ngoài tập sàng lọc vẫn giữ nguyên trong model.

Test lõi dùng fixture nhỏ trong `tests/`, không phải kết quả thực nghiệm:

```bash
cd duong
python -m unittest discover -s tests -v
```

Notebook chọn caption và ground truth từ `mmxai_eval.Protocol` của từng bundle. `xai/queries.csv` vẫn là catalog tham khảo, không được notebook hai bundle sử dụng. `EvidenceSupportRate` coi trạng thái `uncertain` là chưa được hỗ trợ; provenance của cạnh và hỗ trợ thị giác/ngữ nghĩa cần được phân biệt. Đóng góp leave-one-out không được cộng như một phân rã tuyến tính khi scorer phi tuyến.

## 10. VG và PSG trên Colab

Đặt `mmxai_vg_bundle/`, `mmxai_psg_bundle/` và repo `graphml-project3/` tại gốc MyDrive. Hai bundle đã có `huy_extra_cn`; không cần dựng lại nhãn hoặc tải ConceptNet. Chạy theo thứ tự:

1. `notebooks/train_vg_psg_for_xai_colab.ipynb`: huấn luyện VG với SAGE + ConceptNet (`t1_qa_cn_no_sg_sage`) và PSG với GAT + scene graph + ConceptNet (`psg_qa_cn`), mỗi bộ 3 seed; lưu sáu checkpoint và manifest có SHA256.
2. `notebooks/xai_vg_psg_colab.ipynb`: kiểm tra SHA256 và replay hạng gốc, rồi gọi `EdgeContributionExplainer`, `XAIEvaluator` và adapter trong `xai/model.py` trên checkpoint thật. Xóa cạnh được đo trên cùng top-100 ứng viên, random baseline khớp loại cạnh. Xuất CSV/JSON theo dataset, seed, caption và k.

`xai/model.py` dùng chung kiến trúc QA của Huy và định nghĩa hai cấu hình bundle; trọng số, graph và α được lưu riêng theo dataset/seed. Đồ thị XAI chỉ chứa các cạnh được sàng lọc để can thiệp; khi chấm điểm, model vẫn giữ toàn bộ cạnh còn lại. Kết quả là case study chọn theo seed 0, không đại diện cho toàn bộ test. Notebook XAI dọn run, notebook và module demo cũ của Dương trên Drive chỉ sau khi đủ kết quả mới; không xóa bundle hoặc kết quả gốc của Huy.

Hai cấu hình dùng GNN và tập cạnh khác nhau, nên chênh lệch fidelity giữa VG và PSG không thể quy riêng cho dataset. Muốn so sánh tác động dataset, cần thêm một cặp thí nghiệm với cùng kiến trúc và cùng biến thể graph.
