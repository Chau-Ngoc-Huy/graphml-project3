import { useData } from "@/data/DataProvider";
import { METRIC_KEYS } from "@/data/types";
import { modelColor } from "@/lib/models";
import { fmt } from "@/lib/format";
import { cn } from "@/lib/utils";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Delta, ModelDot } from "@/components/common";
import type { ModelRow } from "./useComparison";

/** Side-by-side scorecards or a compact table; best value per metric is highlighted. */
export function MetricsDashboard({ rows, reference, layout }: { rows: ModelRow[]; reference: string; layout: "cards" | "table" }) {
  const ds = useData();
  const ref = rows.find((r) => r.id === reference);
  const best = Object.fromEntries(METRIC_KEYS.map((k) => [k, Math.max(...rows.map((r) => r.metrics[k]))]));
  if (layout === "table")
    return (
      <div className="panel">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Mô hình</TableHead>
              {METRIC_KEYS.map((k) => <TableHead key={k} className="text-right">{k}</TableHead>)}
              <TableHead className="text-right">MedR</TableHead>
              <TableHead className="text-right">Thắng / Hoà / Thua</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r) => {
              const m = ds.modelById.get(r.id)!;
              return (
                <TableRow key={r.id}>
                  <TableCell><div className="flex items-center gap-2"><ModelDot model={m} /><span className="whitespace-nowrap">{m.short}</span>{r.id === reference && <span className="text-[10px] text-muted-foreground">mốc</span>}</div></TableCell>
                  {METRIC_KEYS.map((k) => (
                    <TableCell key={k} className={cn("text-right font-mono", r.metrics[k] === best[k] && "text-primary")}>
                      {fmt(r.metrics[k])}
                      {ref && r.id !== reference && <div><Delta value={r.metrics[k] - ref.metrics[k]} /></div>}
                    </TableCell>
                  ))}
                  <TableCell className="text-right font-mono">{fmt(r.metrics.MedR, 0)}</TableCell>
                  <TableCell className="text-right font-mono text-xs">{r.vsRef ? <><span className="text-good">{r.vsRef.wins}</span> / {r.vsRef.ties} / <span className="text-bad">{r.vsRef.losses}</span></> : "—"}</TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
    );
  return (
    <div className="overflow-x-auto pb-2">
      <div className="grid auto-cols-[minmax(220px,1fr)] grid-flow-col gap-3">
        {rows.map((r) => {
          const m = ds.modelById.get(r.id)!;
          const total = r.vsRef ? r.vsRef.wins + r.vsRef.ties + r.vsRef.losses : 0;
          return (
            <div key={r.id} className={cn("flex flex-col gap-3 rounded-lg border bg-surface p-4", r.id === reference && "border-dashed")}>
              <div className="flex items-center gap-2"><ModelDot model={m} size={9} /><span className="truncate text-[13px] font-medium">{m.short}</span>{r.id === reference && <span className="ml-auto text-[10px] uppercase tracking-wider text-muted-foreground">mốc</span>}</div>
              <div>
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground">MRR</div>
                <div className="flex items-baseline gap-2">
                  <span className={cn("font-mono text-2xl tabular", r.metrics.MRR === best.MRR && "text-primary")} style={{ color: r.metrics.MRR === best.MRR ? undefined : modelColor(m) }}>{fmt(r.metrics.MRR)}</span>
                  {ref && r.id !== reference && <Delta value={r.metrics.MRR - ref.metrics.MRR} />}
                </div>
              </div>
              <div className="grid grid-cols-3 gap-2 text-[11px]">
                {(["R@1", "R@5", "R@10"] as const).map((k) => (
                  <div key={k}><div className="text-muted-foreground">{k}</div><div className={cn("font-mono tabular", r.metrics[k] === best[k] && "text-primary")}>{fmt(r.metrics[k], 1)}</div></div>
                ))}
              </div>
              {r.vsRef && total > 0 && (
                <div className="space-y-1">
                  <div className="flex h-1.5 overflow-hidden rounded-full">
                    <div className="bg-good" style={{ width: `${(100 * r.vsRef.wins) / total}%` }} />
                    <div className="bg-muted-foreground/30" style={{ width: `${(100 * r.vsRef.ties) / total}%` }} />
                    <div className="bg-bad" style={{ width: `${(100 * r.vsRef.losses) / total}%` }} />
                  </div>
                  <div className="flex justify-between font-mono text-[10px] text-muted-foreground"><span className="text-good">{r.vsRef.wins} tốt hơn</span><span className="text-bad">{r.vsRef.losses} kém hơn</span></div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
