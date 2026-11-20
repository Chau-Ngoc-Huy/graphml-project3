import { useMemo } from "react";
import { ArrowRight, CheckCircle2, CircleDashed, CircleDot } from "lucide-react";
import { useData } from "@/data/DataProvider";
import { useApp } from "@/state/AppState";
import { PROBLEM, STAGES, TEAM } from "@/content/story";
import { ADAPTER_ID, BASELINE_ID, BEST_ID, SHUFFLE_ID, modelColor } from "@/lib/models";
import { fmt, fmtInt } from "@/lib/format";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { FadeIn, InfoTip, Section, SourceNote, Stat } from "@/components/common";
import { MetricBars } from "@/components/charts/MetricBars";

export default function ResearchOverview() {
  const ds = useData();
  const { go } = useApp();
  const m = ds.modelById;
  const clip = m.get(BASELINE_ID)!, best = m.get(BEST_ID)!, ad = m.get(ADAPTER_ID)!, shuf = m.get(SHUFFLE_ID)!, adamf = m.get("clip_adamf")!, lex = m.get("clip_lex")!;
  const full = (id: string) => m.get(id)!.official.full.MRR!;
  const g = (mm: typeof best, grp: "direct" | "indirect") => mm.official.groups1k[grp]?.MRR?.mean ?? NaN;
  const paired = ds.results.paired;

  const ladder = useMemo(
    () => [clip, lex, adamf, ad, shuf, best].map((x) => ({ name: x.short, value: x.official.full.MRR!.mean, std: x.official.full.MRR!.std, color: modelColor(x) })),
    [clip, lex, adamf, ad, shuf, best],
  );

  const findings = [
    {
      title: "Đồ thị có đóng góp thật, nhưng nhỏ",
      body: `Cấu hình chính hơn CLIP + Adapter ${fmt(full(BEST_ID).mean - full(ADAPTER_ID).mean)} MRR toàn pool (3 seed), và hơn CLIP ${fmt(full(BEST_ID).mean - full(BASELINE_ID).mean)} MRR.`,
      to: "results" as const,
    },
    {
      title: "Cấu trúc cạnh là yếu tố quyết định",
      body: `Xáo trộn cạnh làm MRR tụt xuống ${fmt(full(SHUFFLE_ID).mean)}, thấp hơn cả Adapter (${fmt(full(ADAPTER_ID).mean)}): mô hình dùng đúng nội dung cạnh, không chỉ nhờ thêm tham số.`,
      to: "results" as const,
    },
    {
      title: "Lợi ích nằm ở truy vấn trực tiếp",
      body: `Trên gallery 1K, nhóm direct tăng ${fmt(g(best, "direct") - g(ad, "direct"))} MRR so với Adapter; nhóm indirect chỉ tăng ${fmt(g(best, "indirect") - g(ad, "indirect"))}. Thông tin có ích là “ảnh chứa đối tượng nào”, chưa phải suy luận quan hệ.`,
      to: "compare" as const,
    },
    {
      title: "Hai hướng độc lập cùng kết luận",
      body: `AdaMF-MAT (MMKG, 2024) chỉ hơn CLIP ${fmt(full("clip_adamf").mean - full(BASELINE_ID).mean)} MRR và vẫn kém baseline khớp tên không dùng quan hệ (${fmt(full("clip_lex").mean)}).`,
      to: "compare" as const,
    },
  ];

  return (
    <div className="mx-auto max-w-[1400px] space-y-14 px-4 py-8 md:px-8">
      {/* Hero */}
      <section className="relative overflow-hidden rounded-xl border bg-surface">
        <div className="grid-bg pointer-events-none absolute inset-0 opacity-40 [mask-image:radial-gradient(ellipse_at_top_right,black,transparent_70%)]" />
        <div className="relative grid gap-10 p-6 md:p-10 xl:grid-cols-[1.15fr_1fr]">
          <FadeIn className="space-y-5">
            <div className="eyebrow">Project 3 · Học máy với dữ liệu đồ thị</div>
            <h1 className="text-balance text-3xl font-semibold leading-tight tracking-tight md:text-[40px]">
              {PROBLEM.title}
            </h1>
            <p className="max-w-[60ch] text-[15px] leading-relaxed text-muted-foreground">{PROBLEM.oneLiner}</p>
            <div className="flex flex-wrap gap-2 pt-1">
              <Button onClick={() => go("playground")}>Thử một truy vấn <ArrowRight /></Button>
              <Button variant="outline" onClick={() => go("journey")}>Xem pipeline</Button>
            </div>
            <div className="flex flex-wrap gap-x-6 gap-y-2 pt-3 text-xs text-muted-foreground">
              <span><span className="font-mono text-foreground">{fmtInt(ds.meta.dataset.images_encoded)}</span> ảnh VG ∩ COCO</span>
              <span><span className="font-mono text-foreground">{fmtInt(ds.meta.dataset.captions)}</span> caption</span>
              <span><span className="font-mono text-foreground">{ds.models.length}</span> cấu hình mô hình</span>
              <span><span className="font-mono text-foreground">{fmtInt(ds.meta.dataset.test_queries)}</span> truy vấn test</span>
            </div>
          </FadeIn>
          <FadeIn delay={0.08}>
            <div className="rounded-lg border bg-background/60 p-5">
              <div className="mb-4 flex items-center justify-between">
                <div className="eyebrow">MRR toàn pool test (2 138 ảnh)</div>
                <InfoTip text="Mean Reciprocal Rank của ảnh đúng, %. Số chính thức: 3 seed (Huy) hoặc 3 fold / 1 seed (Khánh, CLIP tất định). Thanh lỗi = ± std." />
              </div>
              <MetricBars data={ladder} domain={[58, 66]} reference={clip.official.full.MRR!.mean} referenceLabel="CLIP" height={230} />
              <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
                Bậc thang từ CLIP đóng băng tới cấu hình đồ thị tốt nhất. Thanh “Shuffle” là cùng mô hình nhưng xáo trộn cạnh.
              </p>
            </div>
          </FadeIn>
        </div>
      </section>

      {/* Problem → question */}
      <section className="grid gap-8 md:grid-cols-[1fr_1fr]">
        <div className="space-y-3">
          <div className="eyebrow">Vấn đề</div>
          <p className="text-[15px] leading-relaxed">{PROBLEM.problem}</p>
        </div>
        <div className="space-y-3 border-l-2 border-flow/60 pl-5">
          <div className="eyebrow text-flow">Câu hỏi nghiên cứu</div>
          <p className="text-[15px] leading-relaxed">{PROBLEM.question}</p>
        </div>
      </section>

      {/* Headline numbers */}
      <section className="grid grid-cols-2 gap-x-8 gap-y-8 border-y py-8 md:grid-cols-4">
        <Stat label="CLIP (mốc)" value={fmt(full(BASELINE_ID).mean)} sub="MRR toàn pool · tất định" />
        <Stat label="Cấu hình chính" tone="primary" value={fmt(full(BEST_ID).mean)} sub={`± ${fmt(full(BEST_ID).std)} · ${best.short}`} hint={best.desc} />
        <Stat label="Δ so với Adapter" tone="good" value={`+${fmt(full(BEST_ID).mean - full(ADAPTER_ID).mean)}`} sub="phần đóng góp của đồ thị" hint="Adapter = cùng mô hình nhưng tắt đồ thị (no_edge). Đây là mốc đúng để đo đóng góp của đồ thị." />
        <Stat label="Xáo trộn cạnh" tone="bad" value={fmt(full(SHUFFLE_ID).mean)} sub={`thấp hơn Adapter ${fmt(full(ADAPTER_ID).mean - full(SHUFFLE_ID).mean)}`} />
      </section>

      {/* Findings */}
      <Section title="Phát hiện chính" description="Mỗi nhận định đi kèm số liệu tính từ kết quả thật; bấm để xem bằng chứng.">
        <div className="grid gap-px overflow-hidden rounded-lg border bg-border md:grid-cols-2">
          {findings.map((f, i) => (
            <button key={f.title} onClick={() => go(f.to)} className="group flex flex-col gap-2 bg-surface p-5 text-left transition-colors hover:bg-surface-raised">
              <div className="flex items-center gap-2">
                <span className="font-mono text-[11px] text-flow">0{i + 1}</span>
                <h3 className="text-sm font-semibold">{f.title}</h3>
                <ArrowRight className="ml-auto h-3.5 w-3.5 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
              </div>
              <p className="text-[13px] leading-relaxed text-muted-foreground">{f.body}</p>
            </button>
          ))}
        </div>
        <SourceNote>
          MRR toàn pool: huy/results/*kaggle-3seed.json (3 seed), khanh/results/protocol_coco_*.json · nhóm direct/indirect: gallery 1K, mean 3 seed ·
          {paired["t1_qa_cn_no_sg_sage"] ? " kiểm định ghép cặp: khanh/results/vs_huy_paired.json" : ""}
        </SourceNote>
      </Section>

      {/* Pipeline strip */}
      <Section title="Pipeline trong một dòng" description="Truy vấn đi qua 8 bước; bấm một bước để mở chi tiết." aside={<Button variant="ghost" size="sm" onClick={() => go("journey")}>Hành trình đầy đủ <ArrowRight /></Button>}>
        <div className="overflow-x-auto pb-2">
          <ol className="flex min-w-[900px] items-stretch">
            {STAGES.map((s, i) => (
              <li key={s.id} className="flex flex-1 items-center">
                <button onClick={() => go("journey")} className="flex-1 rounded-md border bg-surface px-3 py-2.5 text-left transition-colors hover:border-primary/50">
                  <div className="font-mono text-[10px] text-muted-foreground">{String(i + 1).padStart(2, "0")}</div>
                  <div className="truncate text-[13px] font-medium">{s.short}</div>
                  <div className="mt-1"><StatusBadge status={s.status} /></div>
                </button>
                {i < STAGES.length - 1 && <div className="h-px w-4 shrink-0 bg-flow/60" />}
              </li>
            ))}
          </ol>
        </div>
      </Section>

      {/* Team */}
      <Section title="Đóng góp của nhóm" description="Trạng thái theo những gì đã có trong repo nhóm (graphml-project3).">
        <div className="divide-y rounded-lg border">
          {TEAM.map((t) => (
            <div key={t.name} className="grid gap-3 p-4 md:grid-cols-[180px_1fr_auto] md:items-center">
              <div>
                <div className="text-sm font-medium">{t.name}</div>
                <div className="text-xs text-muted-foreground">{t.role}</div>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {t.done.map((d) => <Badge key={d} variant="muted">{d}</Badge>)}
                {"gap" in t && t.gap && <Badge variant="warn">{t.gap}</Badge>}
              </div>
              <StatusBadge status={t.status} />
            </div>
          ))}
        </div>
      </Section>
    </div>
  );
}

export function StatusBadge({ status }: { status: "done" | "partial" | "planned" }) {
  const map = {
    done: { icon: CheckCircle2, label: "Hoàn thành", cls: "text-good" },
    partial: { icon: CircleDot, label: "Một phần", cls: "text-warn" },
    planned: { icon: CircleDashed, label: "Chưa có", cls: "text-muted-foreground" },
  }[status];
  return (
    <span className={cn("inline-flex items-center gap-1 text-[11px]", map.cls)}>
      <map.icon className="h-3.5 w-3.5" />
      {map.label}
    </span>
  );
}
