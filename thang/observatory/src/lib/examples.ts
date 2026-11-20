import type { Dataset } from "@/data/types";
import { ADAPTER_ID, BASELINE_ID, BEST_ID } from "./models";

export interface Example {
  id: string;
  label: string;
  why: string;
  kind: "test" | "free";
  text: string;
  queryIndex?: number;
}

/** Examples are SELECTED from real per-query ranks by fixed rules (no hand-picking of outcomes). */
export function buildExamples(ds: Dataset): Example[] {
  const clip = ds.ranks[BASELINE_ID];
  const best = ds.ranks[BEST_ID];
  const ad = ds.ranks[ADAPTER_ID];
  const ids = Object.keys(ds.ranks);
  const g = ds.queries.group;
  const pick = (score: (i: number) => number | null) => {
    let bi = -1, bs = -Infinity;
    for (let i = 0; i < clip.length; i++) {
      const s = score(i);
      if (s != null && s > bs) [bi, bs] = [i, s];
    }
    return bi;
  };
  const gain = (a: number, b: number) => Math.log2(b) - Math.log2(a);
  const out: Example[] = [];
  const add = (id: string, label: string, why: string, i: number) => i >= 0 && out.push({ id, label, why, kind: "test", text: ds.queries.text[i], queryIndex: i });

  add("rescue-direct", "Graph rescue (direct)", "Rank gold image cải thiện nhiều nhất từ CLIP sang cấu hình chính, nhóm direct.",
    pick((i) => (g[i] === 0 && best[i] === 1 && ad[i] > 1 ? gain(best[i], clip[i]) : null)));
  add("rescue-indirect", "Graph helps (indirect)", "Trường hợp nhóm indirect được cấu hình chính đưa lên rank 1 trong khi Adapter thì không.",
    pick((i) => (g[i] === 1 && best[i] === 1 && ad[i] > 3 ? gain(best[i], ad[i]) : null)));
  add("hurt", "Graph hurts", "Query mà cấu hình chính đẩy gold image xuống nhiều nhất so với CLIP.",
    pick((i) => (clip[i] <= 2 ? gain(clip[i], best[i]) : null)));
  add("all-fail", "All models miss", "Rank tốt nhất trên tất cả model vẫn > 100.",
    pick((i) => { const m = Math.min(...ids.map((k) => ds.ranks[k][i])); return m > 100 ? m : null; }));
  add("easy", "All models hit", "Mọi model đều xếp gold image ở rank 1 — câu mô tả cụ thể, object rõ.",
    pick((i) => (ids.every((k) => ds.ranks[k][i] === 1) ? ds.queries.concepts[i].length : null)));
  const junk = ds.queries.text.findIndex((t) => /no image/i.test(t));
  add("junk", "Noisy COCO caption", "Người gán nhãn không thấy ảnh — mọi model bất lực; một nguồn nhiễu của dữ liệu.", junk);

  out.push(
    { id: "brief", label: "Assignment example", why: "Câu gián tiếp kinh điển: ảnh có thể chỉ ghi 'ball', 'field', 'boy'. Không thuộc tập test → chỉ chạy bước concept matching.", kind: "free", text: "a child playing football" },
    { id: "kq-indirect", label: "24-query set (indirect)", why: "Bộ 24 query: hai object chỉ nối được qua một object thứ ba.", kind: "free", text: ds.results.kqueries.find((q) => q.group === "indirect")?.text ?? "woman holding boy" },
  );
  return out;
}
