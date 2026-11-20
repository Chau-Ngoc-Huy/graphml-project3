# MMXAI Observatory

Trang web trình bày toàn bộ kết quả Project 3 (truy vấn đa phương thức có giải thích bằng đồ thị): bài toán, pipeline, playground chạy truy vấn trên các mô hình, so sánh mô hình, đồ thị truy vấn và kết quả đánh giá.

**Mọi con số đều lấy từ file kết quả thật của nhóm.** `scripts/build_data.py` đọc kết quả của Thắng, Huy, Khánh và kiểm tra chéo: MRR tính lại từ hạng từng truy vấn phải khớp đúng số mỗi người đã báo cáo (15 kiểm tra), nếu lệch thì dừng.

## Chạy

```bash
npm install                 # lần đầu (Node ≥ 20)
npm run data                # sinh public/data/*.json từ ../../huy, ../../khanh, ../kaggle_results
npm run dev                 # http://localhost:5173
npm run build && npm run preview   # bản production trong dist/
```

`npm run data` cần bundle Nhiệm vụ 1 đã giải nén ở `../../huy/data/task1_bundle/` (caption + đối tượng của ảnh test; không đọc đặc trưng). `public/data/` đã được commit nên chỉ cần chạy lại khi có kết quả mới.

## Các trang

| Trang | Làm gì |
|---|---|
| Tổng quan | Bài toán, câu hỏi nghiên cứu, bậc thang MRR, 4 phát hiện chính, đóng góp từng người |
| Hành trình nghiên cứu | Pipeline 8 bước (React Flow) — bấm một bước để xem vào/ra, cách làm và số liệu thật; dòng thời gian các lượt thử của mô hình đồ thị |
| Playground | Nhập câu hoặc chọn caption test, chọn nhiều mô hình, xem hạng ảnh đúng, Δ so với CLIP, độ trễ, chi phí ghi nhận; tab Raw (Monaco) |
| So sánh mô hình | Chỉ số tính trực tiếp từ hạng từng truy vấn theo tập con (tất cả / direct / indirect), ma trận đối đầu, phân bố hạng, truy vấn bất đồng nhất |
| Đồ thị truy vấn | Câu → khái niệm → đối tượng / quan hệ của ảnh đúng → hạng của từng mô hình; bộ 24 truy vấn của Khánh; mô hình × thành phần |
| Kết quả & đánh giá | Bảng chính thức 16 cấu hình (3 seed), ablation, kiểm định ghép cặp, tái lập AdaMF-MAT, bộ 24 truy vấn, thành công / thất bại, trạng thái xAI |

## Cách "chạy" mô hình

Không có mô hình nào chạy trong trình duyệt. Playground có hai engine cùng một interface (`src/execution/types.ts`):

- **Phát lại kết quả thật** (mặc định): với 10 696 caption test, trả về đúng hạng ảnh đúng mà từng mô hình đã đạt trong lượt chạy thật (seed 0, toàn pool 2 138 ảnh). Câu không thuộc tập test: chỉ chạy bước khớp khái niệm trong trình duyệt; các mô hình học trả về "không khả dụng" kèm gợi ý caption test gần nhất — không bịa kết quả.
- **Live API**: gửi câu tới máy chủ MMXAI Lab của Huy (`huy/ui/server.py`, `POST /api/search`). Khi `npm run dev`, Vite chuyển tiếp `/api` tới `127.0.0.1:8765`. Chưa được kiểm thử với máy chủ đang chạy.

## Cấu trúc

```
scripts/build_data.py        kết quả nhóm → public/data/*.json (+ kiểm tra chéo)
src/data/                    kiểu dữ liệu, loader, DataProvider
src/execution/               engine phát lại + engine Live API
src/lib/                     metric (cùng định nghĩa mmxai_eval.py), linker khái niệm, ví dụ chọn theo quy tắc
src/content/story.ts         nội dung tường thuật (số liệu trích từ JOURNAL.md, có ghi nguồn)
src/components/ui/           shadcn/ui (new-york), src/components/charts/ Recharts
src/features/<trang>/        ResearchOverview, ResearchPipeline, ModelPlayground, ModelComparison, QueryGraphExplorer, ExperimentResults
```

Công nghệ: React 18 + TypeScript, Vite, Tailwind CSS, shadcn/ui (Radix), Recharts, React Flow (@xyflow/react), Monaco, Motion, Lucide.

## Giới hạn

- Hạng từng truy vấn chỉ có cho seed 0; số chính thức 3 seed nằm ở trang Kết quả. Hai biến thể GCN chỉ có số chính thức.
- Phần xAI (fidelity / sparsity / validity) chưa có kết quả trên mô hình thật nên trang không hiển thị số cho phần này.
- Ảnh tải trực tiếp từ máy chủ Visual Genome (cần Internet).
