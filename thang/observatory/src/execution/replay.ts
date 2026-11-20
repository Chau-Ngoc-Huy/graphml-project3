import type { Dataset } from "@/data/types";
import { linkConcepts } from "@/lib/linker";
import { BASELINE_ID } from "@/lib/models";
import type { ExecutionEngine, ExecutionResult, ModelRun, RetrievedImage } from "./types";

const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9 ]+/g, " ").replace(/\s+/g, " ").trim();
const tokens = (s: string) => new Set(norm(s).split(" ").filter((t) => t.length > 1));

/**
 * Replay engine: serves the REAL recorded per-query results of every model on the 10 696 test captions
 * (rank of the ground-truth image, seed 0, full test pool). Arbitrary text cannot be scored by the learned models
 * here (no CLIP/GNN in the browser) — those runs are returned as `unavailable`, never invented.
 */
export function createReplayEngine(ds: Dataset): ExecutionEngine {
  const exact = new Map<string, number>();
  ds.queries.text.forEach((t, i) => {
    const k = norm(t);
    if (!exact.has(k)) exact.set(k, i);
  });
  let tokCache: Set<string>[] | null = null;
  const vocab = new Set(ds.vocab);
  const imageObjects = ds.images.map((im) => new Set(im.objects));

  function similar(input: string, k = 5) {
    tokCache ??= ds.queries.text.map(tokens);
    const a = tokens(input);
    if (!a.size) return [];
    const scored: { queryIndex: number; similarity: number }[] = [];
    tokCache.forEach((b, i) => {
      let inter = 0;
      a.forEach((t) => b.has(t) && inter++);
      if (inter) scored.push({ queryIndex: i, similarity: inter / (a.size + b.size - inter) });
    });
    return scored.sort((x, y) => y.similarity - x.similarity).slice(0, k);
  }

  function conceptMatch(concepts: string[], gold?: number): { latencyMs: number; retrieved: RetrievedImage[] } {
    const t0 = performance.now();
    if (!concepts.length) return { latencyMs: performance.now() - t0, retrieved: [] };
    const scored = imageObjects
      .map((objs, i) => {
        const matched = concepts.filter((c) => objs.has(c));
        return { i, matched, score: matched.length / concepts.length };
      })
      .filter((x) => x.score > 0)
      .sort((a, b) => b.score - a.score || b.matched.length - a.matched.length || a.i - b.i)
      .slice(0, 12);
    const retrieved = scored.map((x, r) => ({
      imageIndex: x.i, url: ds.images[x.i].url, label: `VG ${ds.images[x.i].vg}`, score: x.score, rank: r + 1,
      isGold: gold === x.i, matched: x.matched,
    }));
    return { latencyMs: performance.now() - t0, retrieved };
  }

  return {
    id: "replay",
    label: "Phát lại kết quả thật",
    description: "Kết quả ghi sẵn của từng mô hình trên 10 696 caption test (seed 0, toàn pool 2 138 ảnh). Câu tự do: chỉ chạy bước khớp khái niệm trong trình duyệt.",
    async execute(input, modelIds, opts) {
      const t0 = performance.now();
      const key = norm(input);
      let qi = opts?.forceQueryIndex ?? exact.get(key);
      const suggestions = qi == null ? similar(input) : [];
      const resolution = qi != null ? { mode: "test-caption" as const, queryIndex: qi, similarity: 1 } : { mode: "free-text" as const };
      const text = qi != null ? ds.queries.text[qi] : input;
      const linking = linkConcepts(text, vocab);
      const gold = qi != null ? ds.queries.img[qi] : undefined;
      const base = qi != null ? ds.ranks[BASELINE_ID]?.[qi] : undefined;
      const runs: ModelRun[] = modelIds.map((id) => {
        const s = performance.now();
        const m = ds.models.find((x) => x.id === id);
        if (!m) return { modelId: id, status: "error", latencyMs: 0, note: "Không có mô hình này trong dữ liệu" };
        if (qi == null)
          return {
            modelId: id, status: "unavailable", latencyMs: performance.now() - s,
            note: "Câu không thuộc tập test: mô hình học (CLIP/Adapter/GNN) cần backend để chấm. Chọn một caption test hoặc bật Live API.",
          };
        if (!m.hasRanks)
          return { modelId: id, status: "unavailable", latencyMs: performance.now() - s, note: "Chỉ có số chính thức 3 seed, chưa có hạng từng truy vấn." };
        const r = ds.ranks[id][qi];
        return {
          modelId: id, status: "success", latencyMs: performance.now() - s, goldRank: r, baselineRank: base,
          raw: { model: id, query_index: qi, gold_image_vg: ds.images[gold!].vg, gold_rank: r, clip_rank: base, pool: ds.images.length, seed: 0 },
        };
      });
      const cm = conceptMatch(linking.concepts, gold);
      await Promise.resolve();
      return {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`, engine: "replay", input, resolution, linking,
        goldImageIndex: gold, conceptMatch: cm, runs, suggestions, totalMs: performance.now() - t0, finishedAt: Date.now(),
      } satisfies ExecutionResult;
    },
  };
}
