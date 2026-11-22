# Khảo sát UIT-OpenViIC để dựng MMKG tiếng Việt quy mô nhỏ

Ngày khảo sát: 20/11/2026. Tài liệu này bổ sung cho [Plan 2](02_Vietnamese_Multimodal_KG_Plan.md); các thống kê dưới đây được đo trên thư mục `../../open-viic/` tại máy, không phải ước lượng chất lượng nhãn KG.

## 1. Dữ liệu hiện có và giới hạn

| Thành phần | Kết quả kiểm tra tại máy | Ý nghĩa |
|---|---:|---|
| `v1` | 13.100 ảnh, 61.241 caption; train/dev/test = 9.088/2.011/2.001 ảnh | Khớp [trang giới thiệu của nhóm tác giả](https://uit-together.github.io/datasets/UIT-OpenViIC/). Dùng làm bản chuẩn để lấy mẫu. |
| `annotation-full` | Dictionary `file_name -> {captions: [...]}`, cùng 13.100 ảnh và 61.241 caption | Tiện để đọc toàn bộ caption theo ảnh. |
| `v2` | Vẫn 13.100 ảnh, còn 38.007/9.380/9.125 caption theo train/dev/test | Không gộp với `v1`: nội dung đã thay đổi, có ảnh không còn caption. Chọn và ghi rõ một phiên bản. |
| `images/` | 13.300 tệp ảnh; 13.100 ảnh của `v1` đều tìm thấy | Có 200 tệp ngoài annotation; chỉ lấy ảnh có ID trong annotation. |
| `vinvl_vinvl/*.npy` | Tệp NumPy có kiểu object (`descr: '|O'`) | Đây là đặc trưng phụ, không phải nhãn quan hệ. Không cần dùng cho bản KG đầu tiên; cần xác minh nguồn/ID/định dạng trước khi nạp pickle. |

Mỗi ảnh có khoảng 4–5 caption, nhưng file công bố chỉ có `images` và `annotations` (`image_id`, `caption`); **không có objects, bounding boxes, scene relationships, Wikidata IDs, hay giấy phép theo từng ảnh**. Visual Genome thì có objects, attributes, relationships và scene graph tường minh ([trang tải chính thức](https://visualgenome.org/api/v0), [bài báo](https://arxiv.org/abs/1602.07332)). Vì vậy cần **trích quan hệ mới** từ caption/ảnh, không thể đổi định dạng JSON là có ground truth VG.

[Repo phát hành của tác giả](https://github.com/nghiangh/UIT-OpenViIC-dataset/blob/main/README.md) ghi bộ dữ liệu dùng công khai cho mục đích nghiên cứu. [Trang nguồn](https://uit-together.github.io/datasets/UIT-OpenViIC/) nói ảnh được thu thập chủ yếu qua Google/Bing. Chưa thấy giấy phép cho từng file ảnh trong annotation hiện có; không suy ra quyền tái phân phối ảnh từ việc tải được dataset. Thí nghiệm nội bộ dùng đường dẫn/ID ảnh và ghi nguồn; khi chia sẻ repo hoặc data card, chỉ công bố mã, metadata dẫn xuất và hướng dẫn tải từ nguồn gốc cho đến khi xác minh quyền từng ảnh.

### Kiểm tra nhanh khả năng lấy mẫu

Đếm theo **ảnh có ít nhất một caption chứa từ khóa**, không phải số ảnh đã được xác nhận nội dung, và các nhóm có thể chồng lấn. Cách tính: chuyển caption về chữ thường, tìm cụm bằng regex `(?<!\w)<cụm từ>(?!\w)`; riêng nhóm phố/chợ/quán yêu cầu từ bối cảnh và từ hành động/vị trí nằm trong **cùng một caption**. Split tra từ `v1` theo `file_name`:

| Bộ lọc sơ bộ | Số ảnh | Train / dev / test |
|---|---:|---:|
| Tên món cụ thể: phở, bún, bánh mì, cơm, bánh xèo, gỏi cuốn, bánh cuốn, mì Quảng, hủ tiếu, bánh chưng, bánh tét, bánh bao, chè, lẩu, bánh canh | 251 | 169 / 38 / 44 |
| Từ khóa ẩm thực rộng: đồ ăn, món ăn, thức ăn, bữa ăn, quán ăn, nước mắm, món, thịt, rau, nấu ăn | 1.924 | 1.287 / 325 / 312 |
| Bối cảnh phố/chợ/quán/đường/vỉa hè/cửa hàng **và** ít nhất một từ chỉ hành động/vị trí trong cùng caption | 3.591 | 2.337 / 635 / 619 |
| Caption có các cụm `mặc áo/váy/đầm` hoặc `đội nón/mũ` | 4.664 | Chỉ dùng để tìm ứng viên. |
| Caption có các cụm `đi/chạy/lái xe máy`, `đạp xe`, `lái ô tô` | 389 | Chỉ dùng để tìm ứng viên. |

Phép đếm dùng đối sánh chuỗi/ranh giới từ thô trên `annotation-full`, nên có thể có nghĩa khác ngữ cảnh, caption mô tả sai ảnh và ảnh trùng. Với mốc **300 ảnh**, miền *món ăn cụ thể* không đủ ứng viên theo bộ từ khóa này trước kiểm định. Miền **đời sống đường phố Việt Nam** (người, phương tiện, trang phục, hàng quán, không gian) có dư ứng viên và quan hệ đa dạng hơn. Có thể giữ ẩm thực đường phố như một nhánh để đối chiếu với Plan 2. Nếu bắt buộc chỉ làm ẩm thực, dùng tập rộng rồi kiểm tra tính đồng nhất miền và chấp nhận quy mô thực tế có thể dưới 300.

## 2. Các cách tạo graph

| Hướng | Có thể tạo | Khó khăn chính | Vai trò đề nghị |
|---|---|---|---|
| **Caption → scene graph**: từ điển cụm danh từ + luật hành động/vị trí; sau đó LLM trích JSON theo schema cho phần khó | `image depicts concept`, quan hệ *trong ảnh* như `người đội nón lá`, `người đi xe máy`, `bát ở trên bàn` | Đồng tham chiếu (`cô`, `người phụ nữ`), phủ định, câu không rõ chủ thể, caption mâu thuẫn; cần kiểm tra mẫu | **Nguồn chính**, gần định dạng VG và phong phú cho bài toán retrieval/xAI. |
| **Caption → NER → Wikidata** | QID cho món đặc trưng, địa danh, di tích, tổ chức; quan hệ loại (`P31`, `P279`) và một số thuộc tính thật sự có trong Wikidata | NER thông dụng nhận tên riêng tốt hơn danh từ chung như `xe máy`, `nón lá`, `bát`; tên món dễ đa nghĩa; Wikidata không mô tả quan hệ động trong ảnh | **Nguồn bổ sung có chọn lọc**. Dùng noun phrase/lexicon để bắt concept; chỉ link QID khi nhãn, mô tả và ngữ cảnh đều khớp. |
| **Ảnh → detector/VLM → scene graph** | Nhãn đối tượng cho ảnh chưa có caption liên kết, quan hệ được model gợi ý | Caption tiếng Việt và model thị giác có thể lệch miền; nhãn tự động không phải ground truth | Bổ sung có confidence và rà soát mẫu; quan trọng cho protocol truy xuất chỉ có ảnh ứng viên. |
| **Wikidata/Wikimedia làm graph chính** | `is_a`, `part_of`, vị trí địa lý/loại hình, nguồn tri thức độc lập | Ít khớp với đối tượng chung và quan hệ `wears/rides/next_to`; khó gắn 300 ảnh OpenViIC vào các item cụ thể | Phù hợp nếu đổi sang miền món ăn/địa danh có tên rõ; không đủ một mình để dựng scene graph OpenViIC. |

Wikidata phân biệt `instance of` (P31) và `subclass of` (P279) ([tài liệu mô hình dữ liệu](https://www.wikidata.org/wiki/Wikidata:Data_model/en)); chỉ map theo đúng nghĩa nguồn. Ví dụ `has part(s)` (P527) không mặc nhiên là `has_ingredient`, và `image depicts phở` không kéo theo `image depicts thịt bò`. Truy vấn QID có thể dùng [Wikidata Query Service](https://www.mediawiki.org/wiki/Wikidata_Query_Service/User_Manual), nhưng nên cache và lấy một danh sách QID nhỏ đã rà soát.

**Không dùng NER đơn lẻ để tạo triple**: NER chỉ tìm span/loại thực thể, chưa xác định cặp chủ thể–quan hệ–đối tượng. Cần bước tách cụm danh từ, trích quan hệ, chuẩn hóa alias và kiểm tra bằng chứng. Với LLM, yêu cầu `subject`, `predicate`, `object`, `evidence_span`, `caption_id`; chỉ giữ bộ ba có span đúng trong caption và đúng kiểu quan hệ. Không lấy suy diễn từ kiến thức của model làm bằng chứng ảnh.

## 3. Thiết kế khả thi cho pilot

1. **Chọn miền và mẫu 300 ảnh**: ưu tiên “đời sống đường phố”, lấy mẫu phân tầng từ `v1`: khoảng 210 train, 45 dev, 45 test theo split gốc; đảm bảo phủ ít nhất 5 nhóm tình huống (đi xe, trang phục, cầm đồ, mua bán/ăn uống, vị trí không gian). Lưu seed, ID và tần suất nhóm. Không đưa 200 ảnh không có annotation vào mẫu. Kiểm tra ảnh gần trùng và gom cùng split nếu phát hiện.
2. **Tạo vocab 50–150 concept** từ cụm danh từ: người/cô gái, áo, nón lá, xe máy, bàn, hàng quán, món ăn, cầu, đường... Gộp alias theo ngữ cảnh; giữ QID tùy chọn. Với quan hệ trong từng ảnh, nếu phải phân biệt hai người hoặc hai xe cùng loại, tạo mention ID gắn với ảnh; không coi các mention là 100 concept khác nhau.
3. **Trích scene triples theo caption** với một vocabulary nhỏ (`wears`, `rides`, `holds`, `on`, `beside`, `in_front_of`, `sells`, `eats`), giới hạn luật rõ nghĩa. Mỗi cạnh lưu `image_id`, `caption_id`, đoạn bằng chứng, phương pháp và trạng thái kiểm định. Câu nhiều nghĩa hoặc caption bất đồng → `needs_review`. Ví dụ caption “một cô gái đội nón lá ...” hỗ trợ `(cô gái, đội, nón lá)` trong **ảnh đó**; đây không phải khẳng định mọi cô gái đều đội nón lá.
4. **Dựng hai tầng graph**: bảng `scene_triples` cho bộ ba có ngữ cảnh ảnh; `knowledge_triples` cho cạnh tổng quát từ Wikidata hoặc quy tắc tổng hợp **chỉ trên train** với `support_count`/provenance. `image_entity_links` lưu riêng. Báo ba số cạnh riêng, không cộng để đạt mục tiêu 300 triple tri thức. Wikidata chỉ nối các concept đã được xác nhận. Một graph 300 ảnh × 2 quan hệ/ảnh có thể có khoảng 600 *scene triples* trước lọc, nhưng số triple tri thức duy nhất chưa biết và phải đo ở pilot.
5. **Kiểm định nhỏ**: xem ngẫu nhiên 30 ảnh, 50 cạnh và oversample các trường hợp ít confidence; ghi precision theo từng relation. Kiểm tra ảnh thật có đối tượng, quan hệ đúng chiều, span có thật, QID đúng nghĩa, phân phối relation, node cô lập và ảnh trùng. Nếu chất lượng thấp, sửa luật/ontology chung rồi chạy lại.

### Đầu ra nối với mô hình của nhóm

Xuất bảng giàu provenance theo [Plan 2](02_Vietnamese_Multimodal_KG_Plan.md), đồng thời tạo adapter tương thích [`shared/data-contract.md`](../shared/data-contract.md): `vg/objects.json` từ nhãn concept/ảnh, `vg/relationships.json` từ scene triples theo ảnh, `captions.json`, `splits.json`, và embedding ảnh/caption/nhãn **cùng một encoder**. Các file `vg/*` này là định dạng tương thích, cần ghi `source=OpenViIC-derived`, không gọi là annotation Visual Genome. `vinvl_vinvl` là đặc trưng vùng và không thay thế CLIP embedding trong bundle của Thắng.

Baseline nên dùng cùng candidate pool và top-K như nhiệm vụ 1: CLIP, CLIP + khớp nhãn/đối tượng (không quan hệ), CLIP + graph của Huy, CLIP + AdaMF-MAT của Khánh nếu checkpoint có thể chuyển miền. [Kết quả của Khánh](../khanh/results/protocol_coco_full.md) cho thấy baseline khớp tên có thể bằng hoặc hơn AdaMF-MAT trên VG∩COCO, đặc biệt chưa chứng minh được suy luận quan hệ ở truy vấn indirect. [Kết quả của Huy](../huy/results/JOURNAL.md) cũng cho thấy lợi ích chủ yếu ở direct. Vì thế pilot cần thêm **bỏ/xáo cạnh quan hệ**, cùng một bộ truy vấn có relevance độc lập, và so với adapter không graph. Nếu QID/vocab mới làm embedding hoặc checkpoint cũ không tương thích, báo “adaptation/fine-tuning trên train” thay vì transfer trực tiếp.

**Tránh rò rỉ đánh giá:** caption dev/test có thể làm nhãn đánh giá, nhưng không tạo vocab/cạnh thống kê từ chúng. Nếu dùng caption test để dựng graph cho ảnh test, phải ghi đó là protocol *annotation-assisted/oracle*; truy vấn sinh từ chính caption đó sẽ có lợi thế vòng lặp. Protocol chính nên chỉ dùng ảnh test và detector/VLM đã cố định để tạo nhãn ảnh ứng viên, hoặc dùng caption khác hẳn query và báo rõ mức hỗ trợ annotation. Chọn threshold/alpha trên dev. Với truy vấn indirect, kiểm tra thủ công rằng ảnh đúng không chỉ được nhận ra nhờ một nhãn đối tượng xuất hiện nguyên văn.

### Quyết định đề nghị

Làm pilot **30–50 ảnh** trước khi chốt 300 ảnh: đo precision theo relation, số concept khác nhau và số scene/knowledge triple sau chuẩn hóa. Nếu đạt chất lượng, mở rộng lên 300 ảnh phố/chợ/quán; nếu cần giữ ẩm thực làm miền chính, bắt đầu bằng 150–200 ảnh đã xác nhận về món/quán ăn rồi báo thực tế thay vì ép đủ chỉ tiêu. Ưu tiên graph từ caption có bằng chứng, Wikidata là lớp tri thức bổ sung có QID chắc chắn.
