import type { Dataset } from "@/data/types";
import { linkConcepts } from "@/lib/linker";
import type { ExecutionEngine, ExecutionResult, ModelRun, RetrievedImage } from "./types";

/**
 * Live engine: forwards the query to the MMXAI Lab server (huy/ui/server.py, POST /api/search).
 * In dev, vite proxies /api → http://127.0.0.1:8765 (see vite.config.ts); otherwise set the base URL.
 * The server's variants are its own trained runs (runs/vg_qa*), listed by GET /api/variants.
 */
export interface LiveVariant {
  id: string;
  title?: string;
  exp?: string;
}

export async function fetchLiveVariants(baseUrl: string): Promise<LiveVariant[]> {
  const res = await fetch(`${baseUrl}/api/variants`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const j = await res.json();
  const list = Array.isArray(j) ? j : j.variants ?? [];
  return list.map((v: Record<string, unknown>) => ({ id: String(v.id), title: v.title as string | undefined, exp: v.exp as string | undefined }));
}

type Card = Record<string, unknown>;
const toImage = (c: Card, i: number): RetrievedImage => ({
  url: (c.url as string) ?? (c.coco_url as string) ?? undefined,
  label: String(c.vg_id ?? c.id ?? c.row ?? `#${i + 1}`),
  rank: Number(c.final_rank ?? c.rank ?? i + 1),
  isGold: Boolean(c.is_gold),
  detail: { clip: (c.clip as number) ?? null, adapter: (c.adapter as number) ?? null, graph: (c.graph as number) ?? null, fused: (c.fused as number) ?? null },
});

export function createLiveEngine(ds: Dataset, baseUrl: string): ExecutionEngine {
  const vocab = new Set(ds.vocab);
  return {
    id: "live",
    label: "Live API (máy chủ MMXAI Lab)",
    description: `Gửi query tới ${baseUrl || "(cùng origin)"}/api/search — chạy CLIP + Adapter + graph thật trên server.`,
    async execute(input, modelIds) {
      const t0 = performance.now();
      const runs: ModelRun[] = [];
      for (const variant of modelIds) {
        const s = performance.now();
        try {
          const res = await fetch(`${baseUrl}/api/search`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: input, k: 10, variant }),
          });
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const j = await res.json();
          const A = j.variants?.A ?? {};
          runs.push({
            modelId: variant, status: "success", latencyMs: performance.now() - s, goldRank: A.gold_rank ?? undefined,
            retrieved: ((A.final as Card[]) ?? []).map(toImage), raw: j,
          });
        } catch (e) {
          runs.push({ modelId: variant, status: "error", latencyMs: performance.now() - s, note: (e as Error).message });
        }
      }
      return {
        id: `${Date.now()}`, engine: "live", input, resolution: { mode: "free-text" }, linking: linkConcepts(input, vocab),
        conceptMatch: { latencyMs: 0, retrieved: [] }, runs, suggestions: [], totalMs: performance.now() - t0, finishedAt: Date.now(),
      } satisfies ExecutionResult;
    },
  };
}
