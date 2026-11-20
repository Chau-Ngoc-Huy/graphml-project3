import { useMemo } from "react";
import { ArrowDownToLine, ArrowUpFromLine, Cog } from "lucide-react";
import type { Stage } from "@/content/story";
import { useData } from "@/data/DataProvider";
import { useApp } from "@/state/AppState";
import { linkConcepts } from "@/lib/linker";
import { fmt, fmtInt, fmtStat } from "@/lib/format";
import { modelColor } from "@/lib/models";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Input } from "@/components/ui/input";
import { SourceNote } from "@/components/common";
import { MetricBars } from "@/components/charts/MetricBars";
import { GroupedBars } from "@/components/charts/GroupedBars";
import { COLORS } from "@/components/charts/theme";
import { StatusBadge } from "@/features/overview/ResearchOverview";

function List({ icon: Icon, title, items }: { icon: typeof Cog; title: string; items: string[] }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-muted-foreground"><Icon className="h-3.5 w-3.5" />{title}</div>
      <ul className="space-y-1.5">
        {items.map((x) => <li key={x} className="flex gap-2 text-[13px] leading-relaxed"><span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-flow" />{x}</li>)}
      </ul>
    </div>
  );
}

export function PipelineStageDetail({ stage }: { stage: Stage }) {
  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <StatusBadge status={stage.status} />
        </div>
        <h3 className="text-lg font-semibold tracking-tight">{stage.title}</h3>
        <p className="text-[13px] leading-relaxed">{stage.what}</p>
        <p className="border-l-2 border-flow/50 pl-3 text-[13px] leading-relaxed text-muted-foreground"><span className="text-foreground">Vì sao cần: </span>{stage.why}</p>
      </div>
      <div className="grid gap-5 sm:grid-cols-2">
        <List icon={ArrowDownToLine} title="Đầu vào" items={stage.inputs} />
        <List icon={ArrowUpFromLine} title="Đầu ra" items={stage.outputs} />
      </div>
      <List icon={Cog} title="Cách làm" items={stage.method} />
      <div className="border-t pt-5"><StageViz id={stage.id} /></div>
    </div>
  );
}

function StageViz({ id }: { id: string }) {
  const ds = useData();
  const app = useApp();
  const m = ds.modelById;
  const groupCounts = useMemo(() => {
    const c = [0, 0, 0];
    ds.queries.group.forEach((g) => c[g]++);
    return c;
  }, [ds]);

  if (id === "data") {
    const s = ds.meta.dataset;
    return (
      <div className="space-y-3">
        <div className="eyebrow">Kích thước split (ảnh)</div>
        <GroupedBars xKey="split" height={200} data={(["train", "val", "test"] as const).map((k) => ({ split: k, "Pool Karpathy ∩ VG": s.pool[k], "Dùng trong split": s.split[k] }))}
          series={[{ key: "Pool Karpathy ∩ VG", label: "Pool Karpathy ∩ VG", color: COLORS.muted }, { key: "Dùng trong split", label: "Dùng trong split", color: COLORS.primary }]} />
        <SourceNote>results/clip_baseline/split/split.json</SourceNote>
      </div>
    );
  }
  if (id === "clip") {
    const c = m.get("clip")!.official;
    return (
      <div className="space-y-3">
        <div className="eyebrow">Baseline CLIP · text → image và image → text</div>
        <Table>
          <TableHeader><TableRow><TableHead>Giao thức</TableHead><TableHead className="text-right">R@1</TableHead><TableHead className="text-right">R@10</TableHead><TableHead className="text-right">MRR</TableHead></TableRow></TableHeader>
          <TableBody>
            <TableRow><TableCell>text→image · 1K, 3 fold</TableCell><TableCell className="text-right">{fmtStat(c.test1k["R@1"])}</TableCell><TableCell className="text-right">{fmtStat(c.test1k["R@10"])}</TableCell><TableCell className="text-right">{fmtStat(c.test1k.MRR)}</TableCell></TableRow>
            <TableRow><TableCell>text→image · toàn pool</TableCell><TableCell className="text-right">{fmtStat(c.full["R@1"])}</TableCell><TableCell className="text-right">{fmtStat(c.full["R@10"])}</TableCell><TableCell className="text-right">{fmtStat(c.full.MRR)}</TableCell></TableRow>
            <TableRow><TableCell>image→text · 1K, 3 fold</TableCell><TableCell className="text-right">{fmtStat(ds.results.i2t.test1k["R@1"])}</TableCell><TableCell className="text-right">{fmtStat(ds.results.i2t.test1k["R@10"])}</TableCell><TableCell className="text-right">{fmtStat(ds.results.i2t.test1k.MRR)}</TableCell></TableRow>
          </TableBody>
        </Table>
        <SourceNote>results/adamf/protocol_coco_full.json (CLIP) · results/clip_baseline/results/clip_baseline.json</SourceNote>
      </div>
    );
  }
  if (id === "concepts") {
    const text = app.input || ds.queries.text[0];
    const link = linkConcepts(text, new Set(ds.vocab));
    return (
      <div className="space-y-4">
        <div className="eyebrow">Thử ngay: concept linking chạy trong trình duyệt</div>
        <Input value={app.input} placeholder={ds.queries.text[0]} onChange={(e) => app.setInput(e.target.value)} />
        <div className="flex flex-wrap gap-1 rounded-md border bg-background/60 p-3 font-mono text-[13px] leading-7">
          {link.tokens.map((t, i) => (
            <span key={i} className={t.concept ? "rounded bg-flow/15 px-1 text-flow" : t.stop ? "text-muted-foreground/60" : ""}>{t.text}</span>
          ))}
        </div>
        <div className="flex flex-wrap gap-1.5">{link.concepts.length ? link.concepts.map((c) => <Badge key={c} variant="flow">{c}</Badge>) : <span className="text-xs text-muted-foreground">Không có concept nào trong từ vựng 2 000 label.</span>}</div>
        <div className="eyebrow pt-2">Phân bố nhóm trên {fmtInt(ds.meta.dataset.test_queries)} query test</div>
        <div className="flex h-3 overflow-hidden rounded-sm">
          {groupCounts.map((c, i) => <div key={i} style={{ width: `${(100 * c) / ds.queries.group.length}%`, background: [COLORS.primary, COLORS.warn, COLORS.muted][i] }} />)}
        </div>
        <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
          {ds.queries.groups.map((g, i) => <span key={g}><span className="font-mono text-foreground">{fmtInt(groupCounts[i])}</span> {g}</span>)}
        </div>
      </div>
    );
  }
  if (id === "graph") {
    const rows = ["vg_qa", "vg_qa_cn", "vg_qa_cn_no_sg"].map((k) => ({ k, g: m.get(k)!.official.graph }));
    return (
      <div className="space-y-3">
        <div className="eyebrow">Số edge theo loại</div>
        <GroupedBars xKey="cfg" height={220} data={rows.map(({ k, g }) => ({ cfg: m.get(k)!.short, contains: g?.E_contains ?? 0, sg_rel: g?.E_sg_rel ?? 0, ConceptNet: g?.E_concept ?? 0 }))}
          series={[{ key: "contains", label: "ảnh–chứa–label", color: COLORS.primary }, { key: "sg_rel", label: "relation scene graph", color: COLORS.warn }, { key: "ConceptNet", label: "ConceptNet", color: COLORS.flow }]} />
        <SourceNote>run.extra.graph trong results/*kaggle-3seed.json</SourceNote>
      </div>
    );
  }
  if (id === "gnn") {
    const ids = ["vg_qa_cn_gcn", "vg_qa_cn", "vg_qa_cn_sage", "vg_qa_cn_no_sg_gcn", "vg_qa_cn_no_sg", "vg_qa_cn_no_sg_sage"];
    return (
      <div className="space-y-3">
        <div className="eyebrow">GCN · GAT · GraphSAGE — MRR toàn pool, 3 seed</div>
        <MetricBars domain={[63.5, 65.5]} data={ids.map((k) => { const x = m.get(k)!; return { name: x.short, value: x.official.full.MRR!.mean, std: x.official.full.MRR!.std, color: modelColor(x) }; })} />
        <p className="text-xs text-muted-foreground">Chênh giữa ba loại GNN ≤ 0,3 MRR, cỡ 1–2 độ lệch chuẩn: loại GNN không phải yếu tố chính.</p>
      </div>
    );
  }
  if (id === "fusion") {
    const ms = ds.models.filter((x) => x.official.alpha != null);
    return (
      <div className="space-y-3">
        <div className="eyebrow">α* chọn trên val (1 = chỉ CLIP)</div>
        <div className="space-y-1.5">
          {ms.map((x) => (
            <div key={x.id} className="grid grid-cols-[150px_1fr_44px] items-center gap-3 text-xs">
              <span className="truncate text-muted-foreground">{x.short}</span>
              <div className="relative h-1.5 rounded bg-muted"><div className="absolute inset-y-0 left-0 rounded" style={{ width: `${100 * (x.official.alpha ?? 0)}%`, background: modelColor(x) }} /></div>
              <span className="text-right font-mono tabular">{fmt(x.official.alpha, 2)}</span>
            </div>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">QA-Graph dựa nhiều vào graph score (α ≈ 0,35), còn ghép bằng α tuyến tính như AdaMF-MAT hay presence matching thì vẫn dựa chủ yếu vào CLIP (α ≥ 0,75).</p>
      </div>
    );
  }
  if (id === "eval") {
    return (
      <div className="space-y-3">
        <div className="eyebrow">Kiểm tra chéo khi dựng dữ liệu cho trang này</div>
        <div className="grid gap-1 font-mono text-[11px]">
          {ds.meta.checks.filter((c) => c.mrr_from_ranks != null).map((c) => (
            <div key={c.model} className="flex justify-between gap-3 border-b border-border/50 py-1">
              <span className="truncate text-muted-foreground">{ds.modelById.get(c.model)?.short ?? c.model}</span>
              <span className="text-good">{fmt(c.mrr_from_ranks)} = {fmt(c.reported)}</span>
            </div>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">MRR tính lại từ rank từng query khớp đúng số đã báo cáo trong file kết quả; nhóm direct/indirect khớp bảng trong protocol_coco_full.json → mọi file dùng chung thứ tự query.</p>
        <Button variant="outline" size="sm" onClick={() => app.go("results")}>Xem kết quả đầy đủ</Button>
      </div>
    );
  }
  return (
    <div className="space-y-2 text-[13px] leading-relaxed text-muted-foreground">
      <div className="eyebrow">Trạng thái</div>
      <p>Khung trích path, nhiễu edge và metric fidelity / sparsity / validity đã có (<code className="font-mono">src/mmxai/xai</code>) và đã nối với model thật qua <code className="font-mono">src/mmxai/xai_bridge.py</code> (<code className="font-mono">make xai</code>); trang này chưa hiển thị số fidelity cho bước này.</p>
    </div>
  );
}
