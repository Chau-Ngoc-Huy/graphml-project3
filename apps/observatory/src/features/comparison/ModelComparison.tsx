import { useMemo, useState } from "react";
import { Columns3, Rows3, Search, Shuffle } from "lucide-react";
import { useData } from "@/data/DataProvider";
import { useApp } from "@/state/AppState";
import { modelColor } from "@/lib/models";
import { RANK_BINS } from "@/lib/metrics";
import { fmt, fmtDuration } from "@/lib/format";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ImageThumb, ModelDot, PageHeader, RankPill, Section, SourceNote } from "@/components/common";
import { GroupedBars } from "@/components/charts/GroupedBars";
import { ModelSelector } from "@/features/playground/ModelSelector";
import { MetricsDashboard } from "./MetricsDashboard";
import { HeadToHeadMatrix } from "./HeadToHeadMatrix";
import { findDisagreements, useComparison, type Subset } from "./useComparison";

const COST_NOTES: Record<string, string> = {
  clip: "Notebook giao thức đánh giá: trích 19 264 ảnh + 96 381 caption và đánh giá, ~6 phút trên Kaggle T4.",
  clip_adamf: "AdaMF-MAT 104 epoch × 85,8 s ≈ 2,5 giờ (Kaggle T4, 1 seed) — results/adamf/protocol_coco_full.md.",
  adapter_s_adamf: "Dùng lại checkpoint AdaMF-MAT ở trên; adapter huấn luyện trên CPU.",
};

export default function ModelComparison() {
  const ds = useData();
  const app = useApp();
  const [subset, setSubset] = useState<Subset>("all");
  const [layout, setLayout] = useState<"cards" | "table">("cards");
  const ids = app.compareModels.filter((id) => ds.modelById.has(id));
  const { rows, matrix, n } = useComparison(ids, app.reference, subset);
  const [qSel, setQSel] = useState<number | null>(app.queryIndex);
  const disagreements = useMemo(() => findDisagreements(ds.ranks, ids), [ds, ids]);
  const q = qSel ?? disagreements[0]?.q ?? 0;

  const groupData = useMemo(() => (["direct", "indirect"] as const).map((g) => ({ group: g, ...Object.fromEntries(rows.map((r) => [r.id, r.byGroup[g].MRR])) })), [rows]);
  const histData = useMemo(() => RANK_BINS.map((b, i) => ({ bin: b.label, ...Object.fromEntries(rows.map((r) => [r.id, r.hist[i].pct])) })), [rows]);
  const series = rows.map((r) => { const m = ds.modelById.get(r.id)!; return { key: r.id, label: m.short, color: modelColor(m) }; });

  const strengths = rows.filter((r) => r.id !== app.reference).map((r) => {
    const ref = rows.find((x) => x.id === app.reference);
    const dD = ref ? r.byGroup.direct.MRR - ref.byGroup.direct.MRR : 0;
    const dI = ref ? r.byGroup.indirect.MRR - ref.byGroup.indirect.MRR : 0;
    return { id: r.id, dD, dI, r1: ref ? r.metrics["R@1"] - ref.metrics["R@1"] : 0, r10: ref ? r.metrics["R@10"] - ref.metrics["R@10"] : 0 };
  });

  return (
    <div className="mx-auto max-w-[1400px] space-y-10 px-4 py-8 md:px-8">
      <PageHeader eyebrow="So sánh model" title="Cùng query, nhiều model — khác nhau ở đâu?"
        description={`Mọi con số trên trang này được tính trực tiếp từ rank gold image của từng query (seed 0, toàn pool ${ds.images.length} ảnh test), nên đổi tập con hay model là số đổi theo ngay. Số chính thức 3 seed nằm ở trang Kết quả.`} />

      <div className="panel sticky top-0 z-20 flex flex-col gap-3 p-3 backdrop-blur md:flex-row md:items-center">
        <div className="flex-1"><ModelSelector selected={app.compareModels} onChange={app.setCompareModels} /></div>
        <div className="flex flex-wrap items-center gap-2">
          <Select value={app.reference} onValueChange={app.setReference}>
            <SelectTrigger className="h-8 w-[180px] text-xs"><span className="text-muted-foreground">Mốc:</span><SelectValue /></SelectTrigger>
            <SelectContent>{ds.models.filter((m) => m.hasRanks).map((m) => <SelectItem key={m.id} value={m.id}>{m.short}</SelectItem>)}</SelectContent>
          </Select>
          <ToggleGroup type="single" value={subset} onValueChange={(v) => v && setSubset(v as Subset)}>
            <ToggleGroupItem value="all">Tất cả</ToggleGroupItem>
            <ToggleGroupItem value="direct">Direct</ToggleGroupItem>
            <ToggleGroupItem value="indirect">Indirect</ToggleGroupItem>
          </ToggleGroup>
          <ToggleGroup type="single" value={layout} onValueChange={(v) => v && setLayout(v as "cards" | "table")}>
            <ToggleGroupItem value="cards" aria-label="Edge nhau"><Columns3 /></ToggleGroupItem>
            <ToggleGroupItem value="table" aria-label="Bảng gọn"><Rows3 /></ToggleGroupItem>
          </ToggleGroup>
        </div>
      </div>

      <Section title={`Chỉ số trên ${n.toLocaleString("vi-VN")} query`} description="Màu chàm = tốt nhất ở chỉ số đó. Δ và số query tốt hơn / kém hơn so với model mốc, tính trên từng query.">
        <MetricsDashboard rows={rows} reference={app.reference} layout={layout} />
      </Section>

      <div className="grid gap-8 lg:grid-cols-2">
        <Section title="Direct và indirect" description="MRR theo nhóm query. Khoảng cách giữa hai nhóm cho biết model dựa vào “object có trong ảnh” đến mức nào.">
          <div className="panel p-3"><GroupedBars data={groupData} series={series} xKey="group" domain={[50, "auto"]} /></div>
        </Section>
        <Section title="Rank distribution của gold image" description="% query theo rank bin. Model tốt dồn khối lượng về cột “1”.">
          <div className="panel p-3"><GroupedBars data={histData} series={series} xKey="bin" unit="%" /></div>
        </Section>
      </div>

      <Section title="Điểm mạnh, điểm yếu so với mốc" description={`So với ${ds.modelById.get(app.reference)?.short}: Δ MRR theo nhóm và Δ R@1 / R@10 (điểm %).`}>
        <div className="panel">
          <Table>
            <TableHeader><TableRow><TableHead>Model</TableHead><TableHead className="text-right">Δ direct</TableHead><TableHead className="text-right">Δ indirect</TableHead><TableHead className="text-right">Δ R@1</TableHead><TableHead className="text-right">Δ R@10</TableHead><TableHead>Nhận xét</TableHead></TableRow></TableHeader>
            <TableBody>
              {strengths.map((s) => {
                const m = ds.modelById.get(s.id)!;
                const note = s.dD > 1 && s.dI < 0.5 ? "Mạnh ở câu direct, không giúp câu indirect" : s.dD > 0.5 && s.dI > 0.5 ? "Cải thiện cả hai nhóm" : s.dD < 0 && s.dI < 0 ? "Kém mốc ở cả hai nhóm" : s.r1 > 0 && s.r10 < 0 ? "Đẩy top-1 nhưng mất ở đuôi" : "Khác biệt nhỏ";
                return (
                  <TableRow key={s.id}>
                    <TableCell><div className="flex items-center gap-2"><ModelDot model={m} />{m.short}</div></TableCell>
                    {[s.dD, s.dI, s.r1, s.r10].map((v, i) => <TableCell key={i} className={cn("text-right font-mono", v > 0 ? "text-good" : v < 0 ? "text-bad" : "")}>{v > 0 ? "+" : ""}{fmt(v)}</TableCell>)}
                    <TableCell className="text-xs text-muted-foreground">{note}</TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      </Section>

      <Section title="Đối đầu từng cặp" description="Không chỉ trung bình: trên bao nhiêu query một model thật sự tốt hơn model kia.">
        <div className="panel p-4"><HeadToHeadMatrix ids={ids} matrix={matrix} /></div>
      </Section>

      <Section title="Cùng một query, cạnh nhau" description="Chọn query mà các model đang chọn bất đồng nhiều nhất (chênh lệch log rank lớn nhất), hoặc query đang dùng ở Playground."
        aside={
          <div className="flex gap-2">
            {app.queryIndex != null && <Button variant="outline" size="sm" onClick={() => setQSel(app.queryIndex)}><Search />Query ở Playground</Button>}
            <Button variant="outline" size="sm" onClick={() => setQSel(Math.floor(Math.random() * ds.queries.text.length))}><Shuffle />Ngẫu nhiên</Button>
          </div>
        }>
        <div className="flex flex-wrap gap-1.5">
          {disagreements.map((d) => (
            <button key={d.q} onClick={() => setQSel(d.q)} className={cn("max-w-[260px] truncate rounded-md border px-2 py-1 text-left text-xs transition-colors", d.q === q ? "border-flow/60 bg-flow/10 text-foreground" : "bg-surface text-muted-foreground hover:text-foreground")}>
              {ds.queries.text[d.q]}
            </button>
          ))}
        </div>
        <SameQuery q={q} ids={ids} />
      </Section>

      <Section title="Chi phí tính toán đã ghi nhận" description="Từ chính các lượt chạy thật; ô trống nghĩa là chưa đo — trang không ước lượng thay.">
        <div className="panel">
          <Table>
            <TableHeader><TableRow><TableHead>Model</TableHead><TableHead className="text-right">Thời gian (3 seed)</TableHead><TableHead className="text-right">VRAM node</TableHead><TableHead className="text-right">ms / query</TableHead><TableHead>Ghi chú</TableHead></TableRow></TableHeader>
            <TableBody>
              {ids.map((id) => {
                const m = ds.modelById.get(id)!;
                const c = m.official.cost;
                return (
                  <TableRow key={id}>
                    <TableCell><div className="flex items-center gap-2"><ModelDot model={m} />{m.short}</div></TableCell>
                    <TableCell className="text-right font-mono">{fmtDuration(c?.wall_time_s)}</TableCell>
                    <TableCell className="text-right font-mono">{c?.peak_vram_gb != null ? `${fmt(c.peak_vram_gb, 2)} GB` : "—"}</TableCell>
                    <TableCell className="text-right font-mono">{c?.sec_per_sample != null ? fmt(c.sec_per_sample * 1000, 0) : "—"}</TableCell>
                    <TableCell className="text-xs text-muted-foreground">{COST_NOTES[id] ?? (c ? c.device : "Chưa có số đo")}</TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
        <SourceNote>run.wall_time_s / peak_vram_gb / sec_per_sample trong results/*kaggle-3seed.json</SourceNote>
      </Section>
    </div>
  );
}

function SameQuery({ q, ids }: { q: number; ids: string[] }) {
  const ds = useData();
  const gold = ds.images[ds.queries.img[q]];
  const live = ids.filter((id) => ds.ranks[id]);
  const ranks = live.map((id) => ds.ranks[id][q]);
  const best = Math.min(...ranks), worst = Math.max(...ranks);
  return (
    <div className="grid gap-4 lg:grid-cols-[260px_1fr]">
      <div className="panel space-y-2 p-3">
        <div className="text-[13px] leading-relaxed">“{ds.queries.text[q]}”</div>
        <div className="flex gap-1.5"><Badge variant="outline">#{q}</Badge><Badge variant="muted">{ds.queries.groups[ds.queries.group[q]]}</Badge></div>
        <ImageThumb url={gold.url} alt="gold image" className="aspect-[4/3]" />
        <div className="text-[10px] text-muted-foreground">Gold image · VG {gold.vg}</div>
      </div>
      <div className="overflow-x-auto">
        <div className="grid min-w-[480px] gap-1.5">
          {live.map((id, i) => {
            const m = ds.modelById.get(id)!;
            const r = ranks[i];
            const w = (100 * Math.log(r)) / Math.log(ds.images.length);
            return (
              <div key={id} className={cn("grid grid-cols-[150px_64px_1fr] items-center gap-3 rounded-md border px-3 py-2", r === best && "border-good/40 bg-good/5", r === worst && best !== worst && "border-bad/40 bg-bad/5")}>
                <div className="flex items-center gap-2 truncate text-xs"><ModelDot model={m} />{m.short}</div>
                <RankPill rank={r} />
                <div className="relative h-2 rounded bg-muted"><div className="absolute inset-y-0 left-0 rounded" style={{ width: `${Math.max(2, w)}%`, background: modelColor(m) }} /></div>
              </div>
            );
          })}
        </div>
        <p className="mt-2 text-[11px] text-muted-foreground">Thanh dài hơn = gold image bị xếp thấp hơn (thang log, 1 … {ds.images.length}). Viền xanh / đỏ: tốt nhất / kém nhất.</p>
      </div>
    </div>
  );
}
