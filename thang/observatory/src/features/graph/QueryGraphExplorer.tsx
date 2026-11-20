import { useCallback, useEffect, useMemo, useState } from "react";
import { Background, BackgroundVariant, Controls, MiniMap, ReactFlow, ReactFlowProvider, useReactFlow, type Node } from "@xyflow/react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCcw, Shuffle, Waypoints, X } from "lucide-react";
import { useData } from "@/data/DataProvider";
import { useApp } from "@/state/AppState";
import { buildExamples } from "@/lib/examples";
import { FAMILY_LABEL } from "@/lib/models";
import type { Family } from "@/data/types";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { PageHeader } from "@/components/common";
import { GraphNode } from "./GraphNode";
import { GraphNodeDetail } from "./GraphNodeDetail";
import { buildKQueryGraph, buildModelGraph, buildQueryGraph } from "./builders";
import { KIND_STYLE, type GNodeData } from "./types";

const nodeTypes = { g: GraphNode };
type Mode = "query" | "kq" | "models";

export default function QueryGraphExplorer() {
  return (
    <ReactFlowProvider>
      <Explorer />
    </ReactFlowProvider>
  );
}

function Explorer() {
  const ds = useData();
  const app = useApp();
  const rf = useReactFlow();
  const examples = useMemo(() => buildExamples(ds).filter((e) => e.queryIndex != null), [ds]);
  const [mode, setMode] = useState<Mode>("query");
  const [q, setQ] = useState<number>(app.queryIndex ?? examples[0]?.queryIndex ?? 0);
  const [other, setOther] = useState("8");
  const [show, setShow] = useState({ relations: true, models: true, path: true });
  const [kGroup, setKGroup] = useState<"all" | "direct" | "indirect">("indirect");
  const [family, setFamily] = useState("all");
  const [sel, setSel] = useState<Node<GNodeData> | null>(null);

  const g = useMemo(() => {
    if (mode === "kq") return buildKQueryGraph(ds, kGroup);
    if (mode === "models") return buildModelGraph(ds, family);
    return buildQueryGraph(ds, q, app.compareModels, { otherObjects: Number(other), relations: show.relations, models: show.models, path: show.path });
  }, [ds, mode, q, other, show, kGroup, family, app.compareModels]);

  // Highlight neighbourhood of the selected node.
  const { nodes, edges } = useMemo(() => {
    if (!sel) return g;
    const nb = new Set([sel.id]);
    g.edges.forEach((e) => { if (e.source === sel.id) nb.add(e.target); if (e.target === sel.id) nb.add(e.source); });
    return {
      nodes: g.nodes.map((n) => ({ ...n, selected: n.id === sel.id, data: { ...n.data, dim: !nb.has(n.id) } })),
      edges: g.edges.map((e) => (e.source === sel.id || e.target === sel.id ? { ...e, animated: true, style: { ...e.style, stroke: "hsl(236 84% 68%)", strokeWidth: 1.8 } } : { ...e, style: { ...e.style, opacity: 0.25 } })),
    };
  }, [g, sel]);

  useEffect(() => { setSel(null); const t = setTimeout(() => rf.fitView({ padding: 0.08, duration: 300 }), 30); return () => clearTimeout(t); }, [mode, q, kGroup, family, rf]);
  const reset = useCallback(() => { setSel(null); rf.fitView({ padding: 0.15, duration: 300 }); }, [rf]);

  const legendKinds = mode === "query" ? (["query", "concept", "object", "image", "model"] as const) : mode === "kq" ? (["query", "entity", "via"] as const) : (["component", "model"] as const);

  return (
    <div className="mx-auto max-w-[1600px] space-y-5 px-4 py-8 md:px-8">
      <PageHeader eyebrow="Query graph" title="Một query đi qua graph như thế nào"
        description="Graph dựng từ dữ liệu thật: concept của câu, object và relation trong scene graph của gold image, và rank mà từng model đã cho. Đường xanh là execution path: concept khớp được tên object trong ảnh — đúng tín hiệu mà các graph model khai thác." />

      <div className="flex flex-col gap-3 lg:flex-row lg:items-center">
        <Tabs value={mode} onValueChange={(v) => setMode(v as Mode)}>
          <TabsList>
            <TabsTrigger value="query">Query → ảnh</TabsTrigger>
            <TabsTrigger value="kq">Bộ 24 query</TabsTrigger>
            <TabsTrigger value="models">Model × thành phần</TabsTrigger>
          </TabsList>
        </Tabs>
        <div className="flex flex-1 flex-wrap items-center gap-2 lg:justify-end">
          {mode === "query" && (
            <>
              <Select value={String(q)} onValueChange={(v) => setQ(Number(v))}>
                <SelectTrigger className="h-8 w-[260px] text-xs"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {app.queryIndex != null && !examples.some((e) => e.queryIndex === app.queryIndex) && <SelectItem value={String(app.queryIndex)}>Từ Playground: {ds.queries.text[app.queryIndex].slice(0, 40)}…</SelectItem>}
                  {examples.map((e) => <SelectItem key={e.id} value={String(e.queryIndex)}>{e.label}</SelectItem>)}
                  {!examples.some((e) => e.queryIndex === q) && q !== app.queryIndex && <SelectItem value={String(q)}>#{q}</SelectItem>}
                </SelectContent>
              </Select>
              <Button variant="outline" size="sm" onClick={() => setQ(Math.floor(Math.random() * ds.queries.text.length))}><Shuffle />Ngẫu nhiên</Button>
              <ToggleGroup type="single" value={other} onValueChange={(v) => v && setOther(v)} aria-label="Số object khác">
                {["0", "8", "16"].map((v) => <ToggleGroupItem key={v} value={v}>+{v} object</ToggleGroupItem>)}
              </ToggleGroup>
              {(["relations", "models", "path"] as const).map((k) => (
                <label key={k} className="flex cursor-pointer items-center gap-1.5 text-xs text-muted-foreground">
                  <Checkbox checked={show[k]} onCheckedChange={(c) => setShow({ ...show, [k]: !!c })} />
                  {{ relations: "Relation", models: "Model", path: "Execution path" }[k]}
                </label>
              ))}
            </>
          )}
          {mode === "kq" && (
            <ToggleGroup type="single" value={kGroup} onValueChange={(v) => v && setKGroup(v as typeof kGroup)}>
              <ToggleGroupItem value="all">Tất cả</ToggleGroupItem>
              <ToggleGroupItem value="direct">Direct</ToggleGroupItem>
              <ToggleGroupItem value="indirect">Indirect</ToggleGroupItem>
            </ToggleGroup>
          )}
          {mode === "models" && (
            <ToggleGroup type="single" value={family} onValueChange={(v) => v && setFamily(v)}>
              {["all", "baseline", "lexical", "kgc", "adapter", "graph", "ablation"].map((f) => <ToggleGroupItem key={f} value={f}>{f === "all" ? "Tất cả" : FAMILY_LABEL[f as Family]}</ToggleGroupItem>)}
            </ToggleGroup>
          )}
          <Button variant="ghost" size="sm" onClick={reset}><RotateCcw />Đặt lại</Button>
        </div>
      </div>

      {mode === "query" && (
        <div className="flex items-start gap-2 text-[13px]">
          <Waypoints className="mt-0.5 h-4 w-4 shrink-0 text-flow" />
          <span>“{ds.queries.text[q]}”</span>
        </div>
      )}

      <div className="panel relative h-[640px] overflow-hidden">
        <div className="absolute left-3 top-3 z-10 flex flex-wrap gap-3 rounded-md border bg-surface/90 px-2.5 py-1.5 text-[10px] text-muted-foreground backdrop-blur">
          {legendKinds.map((k) => <span key={k} className="flex items-center gap-1"><span className={cn("h-1.5 w-1.5 rounded-full", KIND_STYLE[k].dot)} />{KIND_STYLE[k].label}</span>)}
        </div>
        {!sel && <div className="absolute right-3 top-3 z-10 rounded-md border bg-surface/90 px-2.5 py-1.5 text-[11px] text-muted-foreground backdrop-blur">Bấm một node để xem chi tiết · kéo để di chuyển · cuộn để phóng to</div>}
        <AnimatePresence>
          {sel && (
            <motion.aside key="inspector" initial={{ opacity: 1, x: 16 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 16 }} transition={{ duration: 0.18 }}
              className="absolute bottom-16 right-3 top-3 z-20 w-[min(320px,calc(100%-1.5rem))] overflow-y-auto rounded-lg border bg-surface/95 p-4 shadow-xl backdrop-blur">
              <button onClick={() => setSel(null)} aria-label="Đóng chi tiết" className="absolute right-3 top-3 text-muted-foreground hover:text-foreground"><X className="h-4 w-4" /></button>
              <GraphNodeDetail node={sel} />
            </motion.aside>
          )}
        </AnimatePresence>
        <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView minZoom={0.2} maxZoom={2} nodesConnectable={false}
          onNodeClick={(_, n) => setSel(n as Node<GNodeData>)} onPaneClick={() => setSel(null)} proOptions={{ hideAttribution: true }}>
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="hsl(222 10% 20%)" />
          <Controls showInteractive={false} position="bottom-right" />
          <MiniMap pannable zoomable position="bottom-left" style={{ width: 140, height: 90 }} className="!hidden md:!block" nodeColor={() => "hsl(222 10% 30%)"} maskColor="hsl(222 14% 7% / 0.7)" />
        </ReactFlow>
      </div>
    </div>
  );
}
