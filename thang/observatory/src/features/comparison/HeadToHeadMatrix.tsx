import { useData } from "@/data/DataProvider";
import type { HeadToHead } from "@/lib/metrics";
import { ModelDot } from "@/components/common";

/** Cell (row A, col B) = % of queries where A ranks the gold image strictly better than B. */
export function HeadToHeadMatrix({ ids, matrix }: { ids: string[]; matrix: (HeadToHead | null)[][] }) {
  const ds = useData();
  const live = ids.filter((id) => ds.ranks[id]);
  const color = (p: number) => {
    const t = Math.max(-1, Math.min(1, (p - 50) / 25));
    return t >= 0 ? `hsl(152 60% 52% / ${0.08 + 0.5 * t})` : `hsl(0 72% 64% / ${0.08 + 0.5 * -t})`;
  };
  return (
    <div className="overflow-x-auto">
      <table className="border-separate border-spacing-1 text-[11px] tabular">
        <thead>
          <tr>
            <th />
            {live.map((id) => { const m = ds.modelById.get(id)!; return <th key={id} className="max-w-[80px] px-1 pb-1 text-left font-normal text-muted-foreground"><div className="flex items-center gap-1"><ModelDot model={m} size={6} /><span className="truncate">{m.short}</span></div></th>; })}
          </tr>
        </thead>
        <tbody>
          {live.map((a, i) => {
            const m = ds.modelById.get(a)!;
            return (
              <tr key={a}>
                <th className="whitespace-nowrap pr-2 text-left font-normal text-muted-foreground"><div className="flex items-center gap-1.5"><ModelDot model={m} size={6} />{m.short}</div></th>
                {matrix[i].map((h, j) => {
                  if (!h) return <td key={j} className="h-9 w-16 rounded bg-muted/40" />;
                  const decided = h.wins + h.losses;
                  const p = decided ? (100 * h.wins) / decided : 50;
                  return (
                    <td key={j} className="h-9 w-16 rounded text-center font-mono" style={{ background: color(p) }} title={`${m.short} tốt hơn ${ds.modelById.get(live[j])?.short} ở ${h.wins} truy vấn, kém hơn ở ${h.losses}, hoà ${h.ties}`}>
                      {p.toFixed(0)}%
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="mt-2 text-[11px] text-muted-foreground">Ô (hàng A, cột B): trong các truy vấn hai mô hình xếp khác nhau, tỉ lệ A xếp ảnh đúng cao hơn B. Xanh = A thường thắng.</p>
    </div>
  );
}
