import { useMemo } from "react";
import type { Node } from "@xyflow/react";
import { useData } from "@/data/DataProvider";
import { useApp } from "@/state/AppState";
import type { ImageInfo, KQuery } from "@/data/types";
import { FAMILY_LABEL } from "@/lib/models";
import { fmt, fmtStat } from "@/lib/format";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ImageThumb, ModelDot, RankPill } from "@/components/common";
import { KIND_STYLE, type GNodeData } from "./types";

export function GraphNodeDetail({ node }: { node: Node<GNodeData> | null }) {
  const ds = useData();
  const app = useApp();
  const imgCount = useMemo(() => {
    const c = new Map<string, number>();
    ds.images.forEach((im) => new Set(im.objects).forEach((o) => c.set(o, (c.get(o) ?? 0) + 1)));
    return c;
  }, [ds]);
  if (!node) return <p className="text-[13px] leading-relaxed text-muted-foreground">Chọn một node để xem chi tiết. Kéo để di chuyển, cuộn để phóng to.</p>;
  const d = node.data;
  const p = d.payload ?? {};
  return (
    <div className="space-y-4">
      <div className="space-y-1.5">
        <Badge variant="outline">{KIND_STYLE[d.kind].label}</Badge>
        <div className="text-[15px] font-semibold leading-snug">{d.label}</div>
        {d.sub && <div className="font-mono text-[11px] text-muted-foreground">{d.sub}</div>}
      </div>
      {d.kind === "query" && typeof p.text === "string" && (
        <div className="space-y-2 text-[13px]">
          <p className="leading-relaxed">“{p.text as string}”</p>
          <Button size="sm" variant="outline" onClick={() => { app.setInput(p.text as string); app.setQueryIndex(p.index as number); app.go("playground"); }}>Mở ở Playground</Button>
        </div>
      )}
      {(d.kind === "concept" || d.kind === "object" || d.kind === "entity" || d.kind === "via") && (
        <div className="space-y-1 text-[13px]">
          <div className="flex justify-between"><span className="text-muted-foreground">Có trong từ vựng label (train)</span><span>{ds.vocab.includes(String(p.concept ?? p.object ?? p.entity ?? p.via)) ? "có" : "không"}</span></div>
          <div className="flex justify-between"><span className="text-muted-foreground">Số ảnh test chứa object này</span><span className="font-mono">{imgCount.get(String(p.concept ?? p.object ?? p.entity ?? p.via)) ?? 0}</span></div>
          {d.kind === "concept" && <p className="pt-1 text-xs text-muted-foreground">{p.inImage ? "Concept có trong danh sách object của gold image: tín hiệu “presence matching” trực tiếp mà các model khai thác." : "Concept KHÔNG có trong gold image → caption thuộc nhóm indirect; chỉ relation / đồng nghĩa mới nối được."}</p>}
        </div>
      )}
      {d.kind === "image" && p.image != null && (() => { const im = p.image as ImageInfo; return (
        <div className="space-y-2">
          <ImageThumb url={im.url} alt="ảnh" className="aspect-[4/3]" />
          <div className="flex flex-wrap gap-1">{im.objects.slice(0, 24).map((o) => <span key={o} className="rounded bg-muted px-1 text-[10px] text-muted-foreground">{o}</span>)}</div>
          <div className="space-y-0.5 font-mono text-[10px] text-muted-foreground">{im.relations.slice(0, 8).map(([s, r, o], i) => <div key={i}>{s} —{r}→ {o}</div>)}</div>
        </div>
      ); })()}
      {d.kind === "model" && typeof p.modelId === "string" && (() => { const m = ds.modelById.get(p.modelId as string)!; return (
        <div className="space-y-2 text-[13px]">
          <div className="flex items-center gap-2"><ModelDot model={m} />{m.name}</div>
          <div className="text-xs text-muted-foreground">{FAMILY_LABEL[m.family]}</div>
          <p className="text-xs leading-relaxed">{m.desc}</p>
          {p.rank != null && <div className="flex items-center gap-2 text-xs">Rank gold image cho query này: <RankPill rank={p.rank as number} /></div>}
          <div className="grid grid-cols-2 gap-2 border-t pt-2 text-xs">
            <div><div className="text-muted-foreground">MRR toàn pool</div><div className="font-mono">{fmtStat(m.official.full.MRR)}</div></div>
            <div><div className="text-muted-foreground">MRR gallery 1K</div><div className="font-mono">{fmtStat(m.official.test1k.MRR)}</div></div>
            <div><div className="text-muted-foreground">α*</div><div className="font-mono">{fmt(m.official.alpha, 2)}</div></div>
            <div><div className="text-muted-foreground">Giao thức</div><div>{m.official.protocol}</div></div>
          </div>
        </div>
      ); })()}
      {d.kind === "query" && p.kq != null && (() => { const q = p.kq as KQuery; return (
        <div className="space-y-2 text-[13px]">
          <div className="font-mono text-xs">{q.triple[0]} —{q.triple[1]}→ {q.triple[2]}</div>
          <div className="text-xs text-muted-foreground">{q.positives.length} ảnh liên quan trong VG. {q.group === "indirect" ? "Hai object không có relation trực tiếp, chỉ nối qua object thứ ba:" : "Gold image có đúng bộ ba này trong scene graph."}</div>
          {q.group === "indirect" && <div className="flex flex-wrap gap-1">{q.examples.map((x, i) => <Badge key={i} variant="warn">qua {x.via}</Badge>)}</div>}
          <Button size="sm" variant="outline" onClick={() => { app.setInput(q.text); app.setQueryIndex(null); app.go("playground"); }}>Chạy ở Playground</Button>
        </div>
      ); })()}
      {d.kind === "component" && <p className="text-xs leading-relaxed text-muted-foreground">Các model nối vào thành phần này đều sử dụng nó. So sánh hai model chỉ khác nhau một thành phần là cách nhóm đọc ablation.</p>}
    </div>
  );
}
