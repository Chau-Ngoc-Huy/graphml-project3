import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Braces, Database, History, LayoutGrid, Plug, ScanSearch, Sparkles, TriangleAlert } from "lucide-react";
import { useData } from "@/data/DataProvider";
import { useApp } from "@/state/AppState";
import { buildExamples } from "@/lib/examples";
import { fetchLiveVariants } from "@/execution/live";
import type { ExecutionResult } from "@/execution/types";
import { fmt } from "@/lib/format";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { EmptyState, ImageThumb, PageHeader } from "@/components/common";
import { QueryInput } from "./QueryInput";
import { ModelSelector } from "./ModelSelector";
import { ModelOutputPanel } from "./ModelOutputPanel";
import { RawOutput } from "./RawOutput";
import { useExecution } from "./useExecution";

export default function ModelPlayground() {
  const ds = useData();
  const app = useApp();
  const ex = useExecution();
  const examples = useMemo(() => buildExamples(ds), [ds]);
  const [liveVariants, setLiveVariants] = useState<string[] | null>(null);
  const [liveMsg, setLiveMsg] = useState<string | null>(null);
  const models = ex.engineId === "live" ? liveVariants ?? [] : app.playModels;

  const run = (forceIndex?: number) => {
    const fi = forceIndex ?? (app.queryIndex != null && ds.queries.text[app.queryIndex] === app.input ? app.queryIndex : undefined);
    app.setQueryIndex(fi ?? null);
    ex.run(app.input, models, fi != null ? { forceQueryIndex: fi } : undefined);
  };

  // Re-run the shared query when arriving from another page (query context is preserved across the app).
  useEffect(() => {
    if (!app.input.trim() && examples[0]) {
      const e = examples[0];
      app.setInput(e.text);
      app.setQueryIndex(e.queryIndex ?? null);
      ex.run(e.text, models, e.queryIndex != null ? { forceQueryIndex: e.queryIndex } : undefined);
      return;
    }
    if (app.input.trim()) run(app.queryIndex ?? undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  // Switching / adding models re-executes the SAME input — the query is never lost.
  useEffect(() => {
    if (ex.result && ex.engineId === "replay") ex.run(ex.result.input, app.playModels, ex.result.resolution.mode === "test-caption" ? { forceQueryIndex: ex.result.resolution.queryIndex } : undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [app.playModels]);

  const connectLive = async () => {
    setLiveMsg("Đang kiểm tra…");
    try {
      const v = await fetchLiveVariants(ex.liveUrl.replace(/\/$/, ""));
      setLiveVariants(v.map((x) => x.id));
      setLiveMsg(`Đã kết nối · ${v.length} biến thể`);
    } catch (e) {
      setLiveVariants(null);
      setLiveMsg(`Không kết nối được: ${(e as Error).message}. Máy chủ MMXAI Lab (ui/server.py, cổng 8765) cần đang chạy.`);
    }
  };

  const r = ex.result;
  return (
    <div className="mx-auto max-w-[1400px] space-y-6 px-4 py-8 md:px-8">
      <PageHeader eyebrow="Model Playground" title="Chạy một query trên nhiều model"
        description="Nhập câu mô tả hoặc chọn một caption test. Với 10 696 caption test, mỗi model trả về đúng rank gold image mà nó đã đạt trong lượt chạy thật; free-text query chỉ chạy được bước concept matching trong trình duyệt, trừ khi bật Live API."
        actions={
          <ToggleGroup type="single" value={ex.engineId} onValueChange={(v) => v && ex.setEngineId(v as "replay" | "live")}>
            <ToggleGroupItem value="replay"><Database />Replay (kết quả thật)</ToggleGroupItem>
            <ToggleGroupItem value="live"><Plug />Live API</ToggleGroupItem>
          </ToggleGroup>
        } />

      {ex.engineId === "live" && (
        <div className="panel flex flex-col gap-3 p-4 md:flex-row md:items-center">
          <div className="flex-1 text-[13px] text-muted-foreground">Gửi query tới máy chủ MMXAI Lab (<code className="font-mono">POST /api/search</code>), chạy CLIP + Adapter + graph thật. Để trống URL khi chạy <code className="font-mono">npm run dev</code> (Vite chuyển tiếp /api tới cổng 8765).</div>
          <Input className="md:w-64" placeholder="http://127.0.0.1:8765 (tuỳ chọn)" value={ex.liveUrl} onChange={(e) => ex.setLiveUrl(e.target.value)} />
          <Button variant="outline" onClick={connectLive}>Kiểm tra kết nối</Button>
          {liveMsg && <span className={cn("text-xs", liveVariants ? "text-good" : "text-warn")}>{liveMsg}</span>}
        </div>
      )}

      <div className="panel space-y-4 p-4 md:p-5">
        <QueryInput value={app.input} onChange={(s) => { app.setInput(s); }} onRun={run} running={ex.status === "running"} />
        <div className="flex flex-wrap items-center gap-2">
          <span className="flex items-center gap-1 text-[11px] uppercase tracking-wider text-muted-foreground"><Sparkles className="h-3 w-3 text-flow" />Ví dụ</span>
          {examples.map((e) => (
            <button key={e.id} title={e.why} onClick={() => { app.setInput(e.text); app.setQueryIndex(e.queryIndex ?? null); ex.run(e.text, models, e.queryIndex != null ? { forceQueryIndex: e.queryIndex } : undefined); }}
              className="rounded-md border bg-background px-2 py-1 text-xs text-muted-foreground transition-colors hover:border-flow/50 hover:text-foreground">
              {e.label}
            </button>
          ))}
        </div>
        {ex.engineId === "replay" && (
          <div className="flex flex-col gap-2 border-t border-border/60 pt-3 md:flex-row md:items-center">
            <span className="text-[11px] uppercase tracking-wider text-muted-foreground md:w-24">Model</span>
            <ModelSelector selected={app.playModels} onChange={app.setPlayModels} focus={app.focusModel} onFocus={app.setFocusModel} />
          </div>
        )}
      </div>

      <div className="grid gap-6 xl:grid-cols-[1fr_260px]">
        <div className="min-w-0 space-y-5">
          <AnimatePresence mode="wait">
            {ex.status === "idle" && !r && (
              <motion.div key="idle" initial={{ opacity: 1 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <EmptyState icon={<ScanSearch className="h-6 w-6" />} title="Chưa có query">Chọn một ví dụ ở trên, bấm “Ngẫu nhiên”, hoặc gõ vài từ để tìm caption test.</EmptyState>
              </motion.div>
            )}
            {ex.status === "running" && !r && (
              <motion.div key="run" className="grid gap-4 md:grid-cols-3"><Skeleton className="h-40" /><Skeleton className="h-40" /><Skeleton className="h-40" /></motion.div>
            )}
            {ex.status === "error" && (
              <motion.div key="err" className="flex items-center gap-2 rounded-md border border-bad/40 bg-bad/10 p-3 text-sm text-bad"><TriangleAlert className="h-4 w-4" />{ex.error}</motion.div>
            )}
          </AnimatePresence>
          {r && <ResultView r={r} running={ex.status === "running"} onPick={(i) => { app.setInput(ds.queries.text[i]); app.setQueryIndex(i); ex.run(ds.queries.text[i], models, { forceQueryIndex: i }); }} />}
        </div>

        <aside className="space-y-2">
          <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-muted-foreground"><History className="h-3.5 w-3.5" />Lịch sử phiên</div>
          {ex.history.length === 0 && <p className="text-xs text-muted-foreground">Các query đã chạy sẽ hiện ở đây.</p>}
          {ex.history.map((h) => (
            <button key={h.id} onClick={() => { ex.setResult(h); app.setInput(h.input); app.setQueryIndex(h.resolution.mode === "test-caption" ? h.resolution.queryIndex : null); }}
              className={cn("w-full rounded-md border px-2.5 py-2 text-left text-xs transition-colors hover:bg-accent/50", r?.id === h.id ? "border-primary/50 bg-primary/5" : "bg-surface")}>
              <div className="line-clamp-2">{h.input}</div>
              <div className="mt-1 flex items-center gap-2 text-[10px] text-muted-foreground">
                <Badge variant={h.resolution.mode === "test-caption" ? "flow" : "muted"}>{h.resolution.mode === "test-caption" ? "caption test" : "free-text query"}</Badge>
                {fmt(h.totalMs, 1)} ms
              </div>
            </button>
          ))}
        </aside>
      </div>
    </div>
  );
}

function ResultView({ r, running, onPick }: { r: ExecutionResult; running: boolean; onPick: (i: number) => void }) {
  const ds = useData();
  const app = useApp();
  const gold = r.goldImageIndex != null ? ds.images[r.goldImageIndex] : null;
  const ok = r.runs.filter((x) => x.status === "success");
  const bestRank = ok.length ? Math.min(...ok.map((x) => x.goldRank ?? Infinity)) : null;
  const goldObjects = new Set(gold?.objects ?? []);
  return (
    <div className={cn("space-y-5 transition-opacity", running && "opacity-60")}>
      {/* resolution + understanding */}
      <div className="grid gap-4 lg:grid-cols-[1fr_280px]">
        <div className="panel space-y-3 p-4">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            {r.resolution.mode === "test-caption" ? (
              <>
                <Badge variant="flow">Caption test #{r.resolution.queryIndex}</Badge>
                <Badge variant="outline">nhóm {ds.queries.groups[ds.queries.group[r.resolution.queryIndex]]}</Badge>
                <span className="text-muted-foreground">Gold image biết trước · kết quả phát lại từ lượt chạy thật (seed 0, toàn pool {ds.images.length} ảnh)</span>
              </>
            ) : (
              <>
                <Badge variant="warn">Free-text query</Badge>
                <span className="text-muted-foreground">Không có gold image; model học cần backend. Bên dưới là bước concept matching chạy trong trình duyệt.</span>
              </>
            )}
            <span className="ml-auto font-mono text-[10px] text-muted-foreground">{fmt(r.totalMs, 1)} ms · engine {r.engine}</span>
          </div>
          <div className="flex flex-wrap gap-1 font-mono text-[13px] leading-7">
            {r.linking.tokens.map((t, i) => (
              <span key={i} title={t.concept ? `concept: ${t.concept}` : undefined}
                className={cn(t.concept ? (goldObjects.has(t.concept) || !gold ? "rounded bg-flow/15 px-1 text-flow" : "rounded bg-warn/15 px-1 text-warn") : t.stop ? "text-muted-foreground/50" : "")}>{t.text}</span>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-muted-foreground">
            <span>Concept:</span>
            {r.linking.concepts.length ? r.linking.concepts.map((c) => (
              <Badge key={c} variant={!gold || goldObjects.has(c) ? "flow" : "warn"}>{c}{gold && !goldObjects.has(c) ? " · không có trong ảnh" : ""}</Badge>
            )) : <span>không liên kết được concept nào</span>}
          </div>
          {r.resolution.mode === "free-text" && r.suggestions.length > 0 && (
            <div className="space-y-1 border-t border-border/60 pt-3">
              <div className="text-[11px] uppercase tracking-wider text-muted-foreground">Caption test gần nhất — chọn để xem kết quả thật của mọi model</div>
              {r.suggestions.map((s) => (
                <button key={s.queryIndex} onClick={() => onPick(s.queryIndex)} className="flex w-full items-center gap-2 rounded px-2 py-1 text-left text-[13px] hover:bg-accent">
                  <span className="font-mono text-[10px] text-muted-foreground">{fmt(s.similarity, 2)}</span>
                  <span className="truncate">{ds.queries.text[s.queryIndex]}</span>
                </button>
              ))}
            </div>
          )}
        </div>
        {gold && (
          <div className="panel space-y-2 p-3">
            <div className="flex items-center justify-between text-[11px] text-muted-foreground"><span className="uppercase tracking-wider">Gold image</span><span className="font-mono">VG {gold.vg}</span></div>
            <ImageThumb url={gold.url} alt="gold image" className="aspect-[4/3]" />
            <div className="flex flex-wrap gap-1">
              {gold.objects.slice(0, 14).map((o) => <span key={o} className={cn("rounded px-1 text-[10px]", r.linking.concepts.includes(o) ? "bg-flow/15 text-flow" : "bg-muted text-muted-foreground")}>{o}</span>)}
            </div>
          </div>
        )}
      </div>

      <Tabs defaultValue="models">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <TabsList>
            <TabsTrigger value="models"><LayoutGrid />Model ({r.runs.length})</TabsTrigger>
            <TabsTrigger value="concept"><ScanSearch />Concept matching</TabsTrigger>
            <TabsTrigger value="raw"><Braces />Raw</TabsTrigger>
          </TabsList>
          {bestRank != null && (
            <div className="text-xs text-muted-foreground">
              Tốt nhất <span className="font-mono text-foreground">#{bestRank}</span> ·{" "}
              {ok.filter((x) => x.goldRank === bestRank).map((x) => ds.modelById.get(x.modelId)?.short).join(", ")}
            </div>
          )}
        </div>
        <TabsContent value="models">
          {r.runs.length === 0 ? (
            <EmptyState title="Chưa chọn model">Thêm model bằng bộ chọn ở trên.</EmptyState>
          ) : (
            <div className="grid gap-3 sm:grid-cols-2 2xl:grid-cols-3">
              {r.runs.map((run, i) => <ModelOutputPanel key={run.modelId} run={run} index={i} focused={app.focusModel === run.modelId} onFocus={() => app.setFocusModel(run.modelId)} />)}
            </div>
          )}
        </TabsContent>
        <TabsContent value="concept" className="space-y-3">
          <p className="text-[13px] text-muted-foreground">
            Phần không cần học của pipeline, chạy trong trình duyệt cho bất kỳ câu nào: xếp {ds.images.length} ảnh test theo tỉ lệ concept của câu có trong danh sách object (scene graph) của ảnh.
            Đây <span className="text-foreground">không</span> phải QA-Graph model, chỉ cho thấy tín hiệu “ảnh chứa object nào” mà các model khai thác ({fmt(r.conceptMatch.latencyMs, 1)} ms).
          </p>
          {r.conceptMatch.retrieved.length === 0 ? <EmptyState title="Không ảnh nào concept matching" /> : (
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 2xl:grid-cols-6">
              {r.conceptMatch.retrieved.map((x) => (
                <div key={x.rank} className="space-y-1.5">
                  <ImageThumb url={x.url} alt={x.label} className={cn("aspect-square", x.isGold && "ring-2 ring-good")}
                    badge={<span className="rounded bg-background/80 px-1 font-mono text-[10px]">#{x.rank}</span>} />
                  <div className="flex flex-wrap gap-1">{x.matched?.map((m) => <span key={m} className="rounded bg-flow/15 px-1 text-[10px] text-flow">{m}</span>)}</div>
                  <div className="font-mono text-[10px] text-muted-foreground">{x.label} · {fmt((x.score ?? 0) * 100, 0)}%{x.isGold ? " · gold image" : ""}</div>
                </div>
              ))}
            </div>
          )}
        </TabsContent>
        <TabsContent value="raw">
          <RawOutput value={{ ...r, linking: { concepts: r.linking.concepts }, conceptMatch: { ...r.conceptMatch, retrieved: r.conceptMatch.retrieved.map(({ imageIndex, url, ...x }) => ({ ...x, image: ds.images[imageIndex ?? 0]?.vg })) } }} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
