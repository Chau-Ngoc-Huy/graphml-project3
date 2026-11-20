import { useMemo, useState } from "react";
import { ArrowDown, ArrowUp, FlaskConical } from "lucide-react";
import { useData } from "@/data/DataProvider";
import { useApp } from "@/state/AppState";
import type { ModelInfo } from "@/data/types";
import { ADAPTER_ID, BASELINE_ID, BEST_ID, FAMILY_LABEL, modelColor } from "@/lib/models";
import { fmt, fmtStat } from "@/lib/format";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Delta, ImageThumb, InfoTip, ModelDot, PageHeader, RankPill, Section, SourceNote } from "@/components/common";
import { MetricBars } from "@/components/charts/MetricBars";
import { GroupedBars } from "@/components/charts/GroupedBars";
import { CIPlot } from "@/components/charts/CIPlot";
import { COLORS } from "@/components/charts/theme";

type SortKey = "full" | "test1k" | "direct" | "indirect";

export default function ExperimentResults() {
  return (
    <div className="mx-auto max-w-[1400px] space-y-8 px-4 py-8 md:px-8">
      <PageHeader eyebrow="Kết quả & đánh giá" title="Bằng chứng định lượng"
        description="Số chính thức của từng người, đúng như trong file kết quả của họ. Mỗi bảng ghi nguồn; mỗi biểu đồ trả lời một câu hỏi cụ thể." />
      <Tabs defaultValue="official">
        <div className="overflow-x-auto pb-1">
          <TabsList>
            <TabsTrigger value="official">Bảng chính thức</TabsTrigger>
            <TabsTrigger value="ablation">Ablation graph</TabsTrigger>
            <TabsTrigger value="paired">Paired bootstrap test</TabsTrigger>
            <TabsTrigger value="repro">AdaMF-MAT reproduction</TabsTrigger>
            <TabsTrigger value="kq">Bộ 24 query</TabsTrigger>
            <TabsTrigger value="cases">Thành công / thất bại</TabsTrigger>
            <TabsTrigger value="xai">Explanation</TabsTrigger>
          </TabsList>
        </div>
        <TabsContent value="official"><Official /></TabsContent>
        <TabsContent value="ablation"><Ablation /></TabsContent>
        <TabsContent value="paired"><Paired /></TabsContent>
        <TabsContent value="repro"><Repro /></TabsContent>
        <TabsContent value="kq"><KQueries /></TabsContent>
        <TabsContent value="cases"><Cases /></TabsContent>
        <TabsContent value="xai"><Xai /></TabsContent>
      </Tabs>
    </div>
  );
}

function Official() {
  const ds = useData();
  const [sort, setSort] = useState<SortKey>("full");
  const val = (m: ModelInfo, k: SortKey) => (k === "full" ? m.official.full.MRR?.mean : k === "test1k" ? m.official.test1k.MRR?.mean : m.official.groups1k[k]?.MRR?.mean) ?? -1;
  const rows = useMemo(() => [...ds.models].sort((a, b) => val(b, sort) - val(a, sort)), [ds, sort]);
  const clip = ds.modelById.get(BASELINE_ID)!;
  const H = ({ k, children }: { k: SortKey; children: React.ReactNode }) => (
    <TableHead className="text-right"><button onClick={() => setSort(k)} className={cn("inline-flex items-center gap-1 uppercase", sort === k && "text-foreground")}>{children}{sort === k && <ArrowDown className="h-3 w-3" />}</button></TableHead>
  );
  return (
    <div className="space-y-8">
      <Section title="MRR toàn pool test của mọi cấu hình" description="2 138 ảnh, 10 696 query. Đường đứt = CLIP. Thanh lỗi = ± std qua 3 seed (QA-Graph); AdaMF-MAT và Adapter-S là 1 lượt huấn luyện.">
        <div className="panel p-4">
          <MetricBars domain={[58, 66]} reference={clip.official.full.MRR!.mean} referenceLabel="CLIP"
            data={[...ds.models].sort((a, b) => (b.official.full.MRR?.mean ?? 0) - (a.official.full.MRR?.mean ?? 0)).map((m) => ({ name: m.short, value: m.official.full.MRR!.mean, std: m.official.full.MRR!.std, color: modelColor(m) }))} />
        </div>
      </Section>
      <Section title="Bảng đầy đủ" description="Bấm tiêu đề cột để sắp xếp. Direct / indirect là MRR trên gallery 1K, trung bình 3 seed / 3 fold.">
        <div className="panel">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Cấu hình</TableHead>
                <TableHead>Họ</TableHead>
                <H k="full">MRR toàn pool</H>
                <TableHead className="text-right">R@1 toàn pool</TableHead>
                <H k="test1k">MRR 1K</H>
                <H k="direct">Direct</H>
                <H k="indirect">Indirect</H>
                <TableHead className="text-right">α*</TableHead>
                <TableHead>Giao thức</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((m) => (
                <TableRow key={m.id}>
                  <TableCell><div className="flex items-center gap-2 whitespace-nowrap"><ModelDot model={m} />{m.name}{m.id === BEST_ID && <Badge>chính</Badge>}</div></TableCell>
                  <TableCell className="text-xs text-muted-foreground">{FAMILY_LABEL[m.family]}</TableCell>
                  <TableCell className="whitespace-nowrap text-right font-mono">{fmtStat(m.official.full.MRR)}</TableCell>
                  <TableCell className="whitespace-nowrap text-right font-mono">{fmtStat(m.official.full["R@1"])}</TableCell>
                  <TableCell className="whitespace-nowrap text-right font-mono">{fmtStat(m.official.test1k.MRR)}</TableCell>
                  <TableCell className="text-right font-mono">{fmt(m.official.groups1k.direct?.MRR?.mean)}</TableCell>
                  <TableCell className="text-right font-mono">{fmt(m.official.groups1k.indirect?.MRR?.mean)}</TableCell>
                  <TableCell className="text-right font-mono">{fmt(m.official.alpha, 2)}</TableCell>
                  <TableCell className="max-w-[180px] text-[11px] text-muted-foreground" title={m.official.source}>{m.official.protocol}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
        <SourceNote>{ds.meta.sources.qa_graph} · {ds.meta.sources.adamf}</SourceNote>
      </Section>
    </div>
  );
}

function Ablation() {
  const ds = useData();
  const ids: [string, string][] = [
    [ADAPTER_ID, "Không edge (no_edge)"],
    ["vg_qa_ablation_no_rel", "Chỉ edge contains"],
    ["vg_qa", "contains + relation scene graph"],
    ["vg_qa_cn", "+ ConceptNet"],
    [BEST_ID, "contains + ConceptNet · SAGE"],
    ["vg_qa_cn_shuffle", "Edge shuffle"],
  ];
  const ad = ds.modelById.get(ADAPTER_ID)!.official;
  const data = ids.map(([id, name]) => { const m = ds.modelById.get(id)!; return { name, value: m.official.full.MRR!.mean, std: m.official.full.MRR!.std, color: modelColor(m) }; });
  const grp = ids.map(([id, name]) => { const g = ds.modelById.get(id)!.official.groups1k; return { cfg: name, direct: g.direct?.MRR?.mean ?? 0, indirect: g.indirect?.MRR?.mean ?? 0 }; });
  return (
    <div className="space-y-8">
      <Section title="Cải thiện có đến từ cấu trúc relation không?" description="Cùng model QA-Graph, chỉ thay tập edge. Mốc đúng là “Không edge”: Adapter học chung nhưng graph score bằng 0.">
        <div className="grid gap-6 lg:grid-cols-[1.2fr_1fr]">
          <div className="panel p-4"><MetricBars data={data} domain={[62, 65.5]} reference={ad.full.MRR!.mean} referenceLabel="Không edge" /></div>
          <div className="space-y-3 text-[13px] leading-relaxed">
            <p><span className="font-medium text-good">Bỏ edge → mất phần tăng.</span> Mọi biến thể có edge đều vượt mốc “Không edge” 1,4–1,9 MRR, CI95 không chứa 0.</p>
            <p><span className="font-medium text-bad">Edge shuffle → tụt dưới mốc.</span> Giữ số edge nhưng nối sai, MRR thấp hơn cả khi không có edge. Model dùng nội dung edge, không phải số tham số.</p>
            <p><span className="font-medium text-warn">Relation không phải nguồn tăng.</span> Chỉ edge contains đã đạt gần trọn mức tăng; thêm relation scene graph hay ConceptNet chênh ≤ 0,2 MRR, trong nhiễu.</p>
            <SourceNote>results/JOURNAL.md #9 · *kaggle-3seed.json</SourceNote>
          </div>
        </div>
      </Section>
      <Section title="Phần tăng nằm ở nhóm nào" description="MRR gallery 1K theo nhóm. Khoảng cách direct – indirect mở rộng khi có edge contains.">
        <div className="panel p-4">
          <GroupedBars xKey="cfg" data={grp} domain={[60, "auto"]} height={280} series={[{ key: "direct", label: "direct", color: COLORS.primary }, { key: "indirect", label: "indirect", color: COLORS.warn }]} />
        </div>
      </Section>
    </div>
  );
}

function Paired() {
  const ds = useData();
  const rows = Object.entries(ds.results.paired).filter(([, v]) => v.vs_AdaMF);
  const nameOf = (k: string) => ds.modelById.get(k)?.short ?? k.replace(/\s*\(.+\)$/, "");
  return (
    <div className="space-y-8">
      <Section title="So với AdaMF-MAT, trên cùng 10 696 query" description="Δ MRR = phương pháp − CLIP+AdaMF-MAT, khoảng tin cậy 95 % bootstrap theo query. Phải 0 = tốt hơn AdaMF-MAT. Xanh: CI không chứa 0.">
        <div className="panel p-4"><CIPlot data={rows.map(([k, v]) => ({ name: nameOf(k), delta: v.vs_AdaMF!.all.delta, lo: v.vs_AdaMF!.all.ci95[0], hi: v.vs_AdaMF!.all.ci95[1] }))} /></div>
        <SourceNote>results/adamf/vs_qagraph_paired.json (seed 0, toàn pool)</SourceNote>
      </Section>
      <div className="panel">
        <Table>
          <TableHeader><TableRow><TableHead>Phương pháp</TableHead><TableHead className="text-right">MRR</TableHead><TableHead className="text-right">Δ tất cả [CI95]</TableHead><TableHead className="text-right">Δ direct</TableHead><TableHead className="text-right">Δ indirect</TableHead></TableRow></TableHeader>
          <TableBody>
            {rows.map(([k, v]) => (
              <TableRow key={k}>
                <TableCell>{nameOf(k)}</TableCell>
                <TableCell className="text-right font-mono">{fmt(v.MRR.all)}</TableCell>
                {(["all", "direct", "indirect"] as const).map((g) => (
                  <TableCell key={g} className="text-right font-mono text-xs">
                    <Delta value={v.vs_AdaMF![g].delta} /> <span className="text-muted-foreground">[{fmt(v.vs_AdaMF![g].ci95[0])}, {fmt(v.vs_AdaMF![g].ci95[1])}]</span>
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}

function Repro() {
  const ds = useData();
  const r = ds.results.repro;
  const metrics = ["MRR", "Hits@1", "Hits@3", "Hits@10"] as const;
  return (
    <Section title="AdaMF-MAT (LREC-COLING 2024) reproduction trên hai MMKG chuẩn" description="Script gốc của tác giả, Kaggle 2×T4, 1 000 epoch, 1 seed. Lệch dưới 1 điểm ở mọi chỉ số → reproduction khớp bài báo.">
      <div className="grid gap-6 lg:grid-cols-2">
        {["MKG-W", "MKG-Y"].map((d) => {
          const p = r.find((x) => x.dataset === d && x.who === "paper")!, o = r.find((x) => x.dataset === d && x.who === "ours")!;
          return (
            <div key={d} className="panel space-y-3 p-4">
              <div className="flex items-center justify-between"><span className="text-sm font-medium">{d}</span><span className="text-[11px] text-muted-foreground">%</span></div>
              <GroupedBars xKey="m" height={220} domain={[0, "auto"]} data={metrics.map((m) => ({ m, "Bài báo": p[m], "Reproduction": o[m] }))}
                series={[{ key: "Bài báo", label: "Bài báo", color: COLORS.muted }, { key: "Reproduction", label: "Reproduction", color: "#e87ba4" }]} />
              <div className="flex flex-wrap gap-4 text-xs">{metrics.map((m) => <span key={m} className="text-muted-foreground">{m} <Delta value={o[m] - p[m]} /></span>)}</div>
            </div>
          );
        })}
      </div>
      <SourceNote>results/adamf/adamf_mat_repro.md</SourceNote>
    </Section>
  );
}

function KQueries() {
  const ds = useData();
  const app = useApp();
  const rows = ds.results.level2;
  return (
    <div className="space-y-8">
      <Section title="Bộ 24 query trực tiếp / gián tiếp" description="12 direct: ảnh có đúng bộ ba chủ thể–hành động–object. 12 indirect: hai object chỉ nối được qua object thứ ba. Chạy trên thí nghiệm riêng (gallery 1 133 ảnh VG, 1 seed).">
        <div className="panel">
          <Table>
            <TableHeader><TableRow><TableHead>Nhóm</TableHead><TableHead>Phương pháp</TableHead><TableHead className="text-right">R@1</TableHead><TableHead className="text-right">R@10</TableHead><TableHead className="text-right">MRR</TableHead><TableHead className="text-right">nDCG@10</TableHead></TableRow></TableHeader>
            <TableBody>
              {rows.map((r, i) => (
                <TableRow key={i} className={cn(r.method.includes("presence") && "bg-warn/5")}>
                  <TableCell className="text-xs text-muted-foreground">{r.group}</TableCell>
                  <TableCell>{r.method}</TableCell>
                  <TableCell className="text-right font-mono">{fmt(r["R@1"], 3)}</TableCell>
                  <TableCell className="text-right font-mono">{fmt(r["R@10"], 3)}</TableCell>
                  <TableCell className="text-right font-mono">{fmt(r.MRR, 3)}</TableCell>
                  <TableCell className="text-right font-mono">{fmt(r["nDCG@10"], 3)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
        <p className="text-[13px] leading-relaxed text-muted-foreground">
          Baseline chỉ đếm object có trong ảnh (<span className="text-foreground">presence-only</span>, không relation) vượt AdaMF-MAT rõ rệt. Theo ghi chú thực nghiệm: 12 query / nhóm là quá ít (1 query = 8 % R@1), 1 seed, và bộ query chưa được người kiểm tra ảnh — chưa đủ để kết luận về vai trò của relation.
        </p>
        <SourceNote>results/adamf/level2_vg_own_experiment.md · queries/eval_queries.json</SourceNote>
      </Section>
      <div className="flex flex-wrap gap-1.5">
        {ds.results.kqueries.map((q) => (
          <button key={q.id} onClick={() => { app.setInput(q.text); app.setQueryIndex(null); app.go("playground"); }}
            className="rounded-md border bg-surface px-2 py-1 text-xs text-muted-foreground transition-colors hover:text-foreground">
            <span className={cn("mr-1.5 font-mono", q.group === "indirect" ? "text-warn" : "text-primary")}>{q.id}</span>{q.text}
          </button>
        ))}
      </div>
    </div>
  );
}

function Cases() {
  const ds = useData();
  const app = useApp();
  const { gains, losses } = useMemo(() => {
    const clip = ds.ranks[BASELINE_ID], best = ds.ranks[BEST_ID];
    const items = clip.map((c, q) => ({ q, d: Math.log2(c) - Math.log2(best[q]) }));
    const pick = (arr: typeof items) => { const seen = new Set<number>(); const out: number[] = []; for (const x of arr) { const im = ds.queries.img[x.q]; if (!seen.has(im)) { seen.add(im); out.push(x.q); } if (out.length === 5) break; } return out; };
    return { gains: pick([...items].sort((a, b) => b.d - a.d)), losses: pick([...items].sort((a, b) => a.d - b.d)) };
  }, [ds]);
  const best = ds.modelById.get(BEST_ID)!;
  const Card = ({ q, up }: { q: number; up: boolean }) => {
    const im = ds.images[ds.queries.img[q]];
    const concepts = ds.queries.concepts[q].map((i) => ds.vocab[i]);
    const objs = new Set(im.objects);
    return (
      <div className="grid grid-cols-[96px_1fr] gap-3 rounded-lg border bg-surface p-3">
        <ImageThumb url={im.url} alt="gold image" className="aspect-square" />
        <div className="min-w-0 space-y-1.5">
          <div className="line-clamp-2 text-[13px]">“{ds.queries.text[q]}”</div>
          <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
            <Badge variant="outline">{ds.queries.groups[ds.queries.group[q]]}</Badge>
            <span className="text-muted-foreground">CLIP</span><RankPill rank={ds.ranks[BASELINE_ID][q]} />
            <span className="text-muted-foreground">Adapter</span><RankPill rank={ds.ranks[ADAPTER_ID][q]} />
            <span className="text-muted-foreground">{best.short}</span><RankPill rank={ds.ranks[BEST_ID][q]} />
            {up ? <ArrowUp className="h-3.5 w-3.5 text-good" /> : <ArrowDown className="h-3.5 w-3.5 text-bad" />}
          </div>
          <div className="flex flex-wrap gap-1">{concepts.map((c) => <span key={c} className={cn("rounded px-1 text-[10px]", objs.has(c) ? "bg-flow/15 text-flow" : "bg-warn/15 text-warn")}>{c}</span>)}</div>
          <Button variant="link" size="sm" className="h-auto p-0 text-[11px]" onClick={() => { app.setInput(ds.queries.text[q]); app.setQueryIndex(q); app.go("graph"); }}>Xem path trong graph →</Button>
        </div>
      </div>
    );
  };
  return (
    <div className="space-y-8">
      <p className="max-w-[80ch] text-[13px] leading-relaxed text-muted-foreground">
        Chọn tự động theo một quy tắc cố định: 5 query có rank gold image cải thiện nhiều nhất và 5 query bị đẩy xuống nhiều nhất khi đi từ CLIP sang cấu hình chính (thang log, mỗi ảnh một lần).
        Concept <span className="text-flow">xanh</span> có trong scene graph của ảnh; <span className="text-warn">vàng</span> thì không.
        <InfoTip text="Mẫu hình chung: thành công khi concept của câu trùng tên object được chú thích trong ảnh; thất bại khi chú thích thiếu, dùng từ khác, hoặc object phổ biến khớp nhầm nhiều ảnh." />
      </p>
      <div className="grid gap-8 lg:grid-cols-2">
        <Section title="Graph helps most"><div className="space-y-2">{gains.map((q) => <Card key={q} q={q} up />)}</div></Section>
        <Section title="Graph hurts nhiều nhất"><div className="space-y-2">{losses.map((q) => <Card key={q} q={q} up={false} />)}</div></Section>
      </div>
    </div>
  );
}

function Xai() {
  return (
    <div className="max-w-3xl space-y-4">
      <div className="flex items-center gap-2"><FlaskConical className="h-4 w-4 text-warn" /><span className="text-sm font-medium">Chưa có số đánh giá explanation trên model thật</span></div>
      <p className="text-[13px] leading-relaxed text-muted-foreground">
        Đề yêu cầu fidelity (xoá / nhiễu edge explanation thì điểm phải đổi), validity và sparsity trên ít nhất 10 query. Đã có khung trích path, nhiễu edge và các metric này kèm test, nhưng mới chạy với scorer giả; explain.py có fidelity trên dữ liệu giả.
        Trang này cố ý không hiển thị con số nào cho phần này cho tới khi có kết quả trên cấu hình chính.
      </p>
      <p className="text-[13px] leading-relaxed text-muted-foreground">Trong lúc chờ, trang Query graph cho thấy path concept → object → ảnh mà graph score dựa vào, dựng từ dữ liệu thật.</p>
      <SourceNote>src/mmxai/xai/*, tests/xai/* · src/mmxai/graph/explain.py</SourceNote>
    </div>
  );
}
