import type { MetricKey } from "@/data/types";

export type RankMetrics = Record<MetricKey, number> & { MedR: number; n: number };

/** Same definitions as thang/mmxai_eval.py: rank of the single gold image, 1-based. */
export function rankMetrics(ranks: ArrayLike<number>, idx?: ArrayLike<number>): RankMetrics {
  const n = idx ? idx.length : ranks.length;
  let r1 = 0, r5 = 0, r10 = 0, rr = 0, nd = 0;
  const vals: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const r = ranks[idx ? idx[i] : i];
    vals[i] = r;
    if (r <= 1) r1++;
    if (r <= 5) r5++;
    if (r <= 10) {
      r10++;
      nd += 1 / Math.log2(r + 1);
    }
    rr += 1 / r;
  }
  vals.sort((a, b) => a - b);
  const med = n ? (n % 2 ? vals[(n - 1) / 2] : (vals[n / 2 - 1] + vals[n / 2]) / 2) : NaN;
  const k = n || 1;
  return { "R@1": (100 * r1) / k, "R@5": (100 * r5) / k, "R@10": (100 * r10) / k, MRR: (100 * rr) / k, "nDCG@10": (100 * nd) / k, MedR: med, n };
}

export interface HeadToHead {
  wins: number;
  ties: number;
  losses: number;
  meanRRDelta: number;
  n: number;
}

/** Per-query comparison of model A against reference B (lower rank is better). */
export function headToHead(a: ArrayLike<number>, b: ArrayLike<number>, idx?: ArrayLike<number>): HeadToHead {
  const n = idx ? idx.length : a.length;
  let w = 0, t = 0, l = 0, d = 0;
  for (let i = 0; i < n; i++) {
    const j = idx ? idx[i] : i;
    if (a[j] < b[j]) w++;
    else if (a[j] > b[j]) l++;
    else t++;
    d += 1 / a[j] - 1 / b[j];
  }
  return { wins: w, ties: t, losses: l, meanRRDelta: (100 * d) / (n || 1), n };
}

export const RANK_BINS = [
  { label: "1", lo: 1, hi: 1 },
  { label: "2", lo: 2, hi: 2 },
  { label: "3–5", lo: 3, hi: 5 },
  { label: "6–10", lo: 6, hi: 10 },
  { label: "11–20", lo: 11, hi: 20 },
  { label: "21–100", lo: 21, hi: 100 },
  { label: ">100", lo: 101, hi: Infinity },
];

export function rankHistogram(ranks: ArrayLike<number>, idx?: ArrayLike<number>) {
  const n = idx ? idx.length : ranks.length;
  const counts = RANK_BINS.map(() => 0);
  for (let i = 0; i < n; i++) {
    const r = ranks[idx ? idx[i] : i];
    const k = RANK_BINS.findIndex((b) => r >= b.lo && r <= b.hi);
    counts[k]++;
  }
  return RANK_BINS.map((b, i) => ({ bin: b.label, count: counts[i], pct: (100 * counts[i]) / (n || 1) }));
}
