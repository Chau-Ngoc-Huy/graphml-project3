import { useMemo, useState } from "react";
import { Background, BackgroundVariant, Controls, MarkerType, ReactFlow, type Edge, type Node } from "@xyflow/react";
import { AnimatePresence, motion } from "motion/react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { GRAPH_JOURNEY, STAGES } from "@/content/story";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { PageHeader, Section, SourceNote } from "@/components/common";
import { AXIS, COLORS, GRID, TOOLTIP_STYLE } from "@/components/charts/theme";
import { StageNode, type StageNodeData } from "./StageNode";
import { PipelineStageDetail } from "./PipelineStageDetail";

const nodeTypes = { stage: StageNode };
const COL = 200;

export default function ResearchPipeline() {
  const [sel, setSel] = useState("graph");
  const stage = STAGES.find((s) => s.id === sel)!;

  const { nodes, edges } = useMemo(() => {
    const ns: Node<StageNodeData>[] = STAGES.map((s, i) => ({
      id: s.id, type: "stage", position: { x: (i % 4) * COL, y: i < 4 ? 0 : 150 },
      data: { index: i + 1, title: s.short, owner: s.owner, status: s.status, selected: s.id === sel },
    }));
    ns.push({ id: "adamf", type: "stage", position: { x: 2 * COL + 20, y: 300 }, data: { index: 0, title: "AdaMF-MAT (MMKG)", owner: "Khánh · nhánh song song", status: "done", selected: false, branch: true } });
    const es: Edge[] = STAGES.filter((s) => s.next).map((s) => ({
      id: `${s.id}-${s.next}`, source: s.id, target: s.next!, animated: s.id === sel || s.next === sel,
      style: { stroke: "hsl(var(--flow))", strokeWidth: s.id === sel || s.next === sel ? 1.8 : 1, opacity: s.id === sel || s.next === sel ? 1 : 0.45 },
      markerEnd: { type: MarkerType.ArrowClosed, color: "hsl(188 86% 53%)", width: 14, height: 14 },
    }));
    es.push(
      { id: "c-adamf", source: "concepts", target: "adamf", style: { stroke: "hsl(var(--muted-foreground))", strokeDasharray: "4 3" } },
      { id: "adamf-f", source: "adamf", target: "fusion", style: { stroke: "hsl(var(--muted-foreground))", strokeDasharray: "4 3" } },
    );
    return { nodes: ns, edges: es };
  }, [sel]);

  return (
    <div className="mx-auto max-w-[1400px] space-y-12 px-4 py-8 md:px-8">
      <PageHeader eyebrow="Hành trình nghiên cứu" title="Từ một câu mô tả đến thứ hạng có thể kiểm chứng"
        description="Pipeline đầy đủ của nhóm. Mỗi bước nhận đầu ra của bước trước; chọn một bước để xem nó làm gì, vì sao cần, dữ liệu vào/ra và số liệu thật của bước đó." />

      <div className="grid gap-6 xl:grid-cols-[1.25fr_1fr]">
        <div className="panel relative h-[460px] overflow-hidden">
          <div className="absolute left-4 top-3 z-10 flex gap-3 text-[11px] text-muted-foreground">
            <span className="flex items-center gap-1"><span className="h-1.5 w-1.5 rounded-full bg-good" />Hoàn thành</span>
            <span className="flex items-center gap-1"><span className="h-1.5 w-1.5 rounded-full bg-warn" />Một phần</span>
            <span className="flex items-center gap-1"><span className="h-px w-4 border-t border-dashed border-muted-foreground" />Nhánh song song</span>
          </div>
          <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView fitViewOptions={{ padding: 0.18 }} nodesDraggable={false} nodesConnectable={false}
            onNodeClick={(_, n) => n.id !== "adamf" ? setSel(n.id) : setSel("gnn")} proOptions={{ hideAttribution: true }} minZoom={0.4} maxZoom={1.6}>
            <Background variant={BackgroundVariant.Dots} gap={18} size={1} color="hsl(222 10% 22%)" />
            <Controls showInteractive={false} position="bottom-right" />
          </ReactFlow>
        </div>
        <div className="panel min-h-[460px] p-5 md:p-6">
          <div className="mb-4 flex flex-wrap gap-1">
            {STAGES.map((s, i) => (
              <button key={s.id} onClick={() => setSel(s.id)} className={cn("rounded px-2 py-1 font-mono text-[11px] transition-colors", s.id === sel ? "bg-primary/15 text-primary" : "text-muted-foreground hover:text-foreground")}>
                {String(i + 1).padStart(2, "0")}
              </button>
            ))}
          </div>
          <AnimatePresence mode="wait">
            <motion.div key={sel} initial={{ opacity: 1, x: 8 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -8 }} transition={{ duration: 0.18 }}>
              <PipelineStageDetail stage={stage} />
            </motion.div>
          </AnimatePresence>
        </div>
      </div>

      <Section title="Mô hình đồ thị đã thay đổi thế nào qua từng lượt thử"
        description="Nhật ký thực nghiệm của Huy. Đường xanh là mô hình đồ thị, đường xám là mốc so sánh đúng ở lượt đó. Lượt #3 cho thấy mức tăng ban đầu không đến từ đồ thị — một kết quả âm được giữ lại.">
        <div className="grid gap-6 lg:grid-cols-[1fr_1.1fr]">
          <div className="panel p-4">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={GRAPH_JOURNEY.map((j) => ({ ...j }))} margin={{ top: 10, right: 16, bottom: 0, left: -12 }}>
                <CartesianGrid vertical={false} {...GRID} />
                <XAxis dataKey="id" {...AXIS} tickLine={false} />
                <YAxis domain={[60, 66]} {...AXIS} tickLine={false} axisLine={false} />
                <Tooltip {...TOOLTIP_STYLE} formatter={(v: number, n) => [v.toFixed(2), n]} />
                <Line type="linear" dataKey="ref" name="Mốc so sánh" stroke={COLORS.muted} strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
                <Line type="linear" dataKey="graph" name="Mô hình đồ thị" stroke={COLORS.primary} strokeWidth={2} dot={{ r: 3.5 }} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
            <SourceNote>huy/results/JOURNAL.md · MRR % toàn pool test, seed 0 (lượt #9: 3 seed)</SourceNote>
          </div>
          <ol className="relative space-y-4 border-l pl-5">
            {GRAPH_JOURNEY.map((j) => (
              <li key={j.id} className="relative">
                <span className="absolute -left-[25px] top-1.5 h-2 w-2 rounded-full border-2 border-background bg-primary" />
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-mono text-[11px] text-muted-foreground">{j.id} · {j.date}</span>
                  <span className="text-[13px] font-medium">{j.title}</span>
                  <Badge variant={j.verdict === "bác bỏ" || j.verdict === "không giúp" || j.verdict === "quan hệ không giúp" ? "bad" : j.verdict === "kết luận" || j.verdict === "có tác dụng" ? "good" : "muted"}>{j.verdict}</Badge>
                </div>
                <p className="mt-1 text-[13px] leading-relaxed text-muted-foreground">{j.finding}</p>
              </li>
            ))}
          </ol>
        </div>
      </Section>
    </div>
  );
}
