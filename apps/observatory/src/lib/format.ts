import type { Stat } from "@/data/types";

export const fmt = (x: number | null | undefined, d = 2) => (x == null || Number.isNaN(x) ? "—" : x.toFixed(d));
export const fmtInt = (x: number | null | undefined) => (x == null ? "—" : x.toLocaleString("vi-VN"));
export const fmtSigned = (x: number, d = 2) => `${x > 0 ? "+" : x < 0 ? "−" : "±"}${Math.abs(x).toFixed(d)}`;
export const fmtStat = (s?: Stat, d = 2) => (!s ? "—" : s.std == null ? fmt(s.mean, d) : `${fmt(s.mean, d)} ± ${fmt(s.std, d)}`);
export function fmtDuration(sec?: number | null) {
  if (sec == null) return "—";
  if (sec < 60) return `${sec.toFixed(0)} s`;
  const m = sec / 60;
  return m < 60 ? `${m.toFixed(1)} phút` : `${(m / 60).toFixed(1)} giờ`;
}
