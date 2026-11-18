import { motion } from "motion/react";

/** Position of the gold image in the full pool on a log scale (1 … pool). Left = better. */
export function RankMeter({ rank, base, pool, color }: { rank: number; base?: number; pool: number; color: string }) {
  const pos = (r: number) => (100 * Math.log(r)) / Math.log(pool);
  return (
    <div className="relative h-5">
      <div className="absolute inset-x-0 top-1/2 h-px -translate-y-1/2 bg-border" />
      {[1, 10, 100, 1000].map((t) => (
        <div key={t} className="absolute top-1/2 h-2 w-px -translate-y-1/2 bg-border" style={{ left: `${pos(t)}%` }}>
          <span className="absolute left-1/2 top-2.5 -translate-x-1/2 font-mono text-[9px] text-muted-foreground/70">{t}</span>
        </div>
      ))}
      {base != null && base !== rank && (
        <div className="absolute top-1/2 h-px -translate-y-1/2 bg-muted-foreground/50" style={{ left: `${Math.min(pos(base), pos(rank))}%`, width: `${Math.abs(pos(base) - pos(rank))}%` }} />
      )}
      {base != null && <div className="absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-muted-foreground bg-background" style={{ left: `${pos(base)}%` }} title={`CLIP #${base}`} />}
      <motion.div initial={{ left: base != null ? `${pos(base)}%` : 0 }} animate={{ left: `${pos(rank)}%` }} transition={{ duration: 0.5, ease: [0.2, 0.7, 0.2, 1] }}
        className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-background" style={{ background: color }} />
    </div>
  );
}
