import { useMemo } from "react";
import { useData } from "@/data/DataProvider";
import { headToHead, rankHistogram, rankMetrics, type HeadToHead, type RankMetrics } from "@/lib/metrics";

export type Subset = "all" | "direct" | "indirect";

export interface ModelRow {
  id: string;
  metrics: RankMetrics;
  byGroup: Record<"direct" | "indirect", RankMetrics>;
  vsRef: HeadToHead | null;
  hist: ReturnType<typeof rankHistogram>;
}

/** All comparison numbers are derived from per-query ranks (seed 0, full test pool) — never stored aggregates. */
export function useComparison(ids: string[], reference: string, subset: Subset) {
  const ds = useData();
  const groups = useMemo(() => {
    const d: number[] = [], i: number[] = [], all: number[] = [];
    ds.queries.group.forEach((g, q) => { all.push(q); if (g === 0) d.push(q); if (g === 1) i.push(q); });
    return { all, direct: d, indirect: i };
  }, [ds]);
  const idx = groups[subset];
  const rows = useMemo<ModelRow[]>(() => {
    const ref = ds.ranks[reference];
    return ids.filter((id) => ds.ranks[id]).map((id) => {
      const r = ds.ranks[id];
      return {
        id, metrics: rankMetrics(r, idx),
        byGroup: { direct: rankMetrics(r, groups.direct), indirect: rankMetrics(r, groups.indirect) },
        vsRef: ref && id !== reference ? headToHead(r, ref, idx) : null,
        hist: rankHistogram(r, idx),
      };
    });
  }, [ds, ids, reference, idx, groups]);
  const matrix = useMemo(() => {
    const live = ids.filter((id) => ds.ranks[id]);
    return live.map((a) => live.map((b) => (a === b ? null : headToHead(ds.ranks[a], ds.ranks[b], idx))));
  }, [ds, ids, idx]);
  return { rows, matrix, n: idx.length, groups };
}

/** Test captions where the selected models disagree most (spread of log-rank), from real ranks. */
export function findDisagreements(ranks: Record<string, number[]>, ids: string[], k = 6) {
  const live = ids.filter((id) => ranks[id]);
  if (live.length < 2) return [];
  const n = ranks[live[0]].length;
  const out: { q: number; spread: number }[] = [];
  for (let q = 0; q < n; q++) {
    let lo = Infinity, hi = -Infinity;
    for (const id of live) { const v = Math.log2(ranks[id][q]); if (v < lo) lo = v; if (v > hi) hi = v; }
    out.push({ q, spread: hi - lo });
  }
  return out.sort((a, b) => b.spread - a.spread).slice(0, k);
}
