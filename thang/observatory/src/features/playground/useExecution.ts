import { useCallback, useMemo, useRef, useState } from "react";
import { useData } from "@/data/DataProvider";
import { createReplayEngine } from "@/execution/replay";
import { createLiveEngine } from "@/execution/live";
import type { ExecutionEngine, ExecutionResult } from "@/execution/types";

export type EngineId = "replay" | "live";
export type ExecStatus = "idle" | "running" | "success" | "error";

/** Query execution state: engine choice, status, current result and a session history. */
export function useExecution() {
  const ds = useData();
  const [engineId, setEngineId] = useState<EngineId>("replay");
  const [liveUrl, setLiveUrl] = useState("");
  const [status, setStatus] = useState<ExecStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ExecutionResult | null>(null);
  const [history, setHistory] = useState<ExecutionResult[]>([]);
  const replay = useMemo(() => createReplayEngine(ds), [ds]);
  const live = useMemo(() => createLiveEngine(ds, liveUrl.replace(/\/$/, "")), [ds, liveUrl]);
  const engine: ExecutionEngine = engineId === "live" ? live : replay;
  const seq = useRef(0);

  const run = useCallback(
    async (input: string, modelIds: string[], opts?: { forceQueryIndex?: number }) => {
      if (!input.trim()) return;
      const my = ++seq.current;
      setStatus("running");
      setError(null);
      try {
        const r = await engine.execute(input.trim(), modelIds, opts);
        if (my !== seq.current) return;
        setResult(r);
        setHistory((h) => [r, ...h.filter((x) => x.input !== r.input)].slice(0, 10));
        setStatus("success");
      } catch (e) {
        if (my !== seq.current) return;
        setError((e as Error).message);
        setStatus("error");
      }
    },
    [engine],
  );

  return { engine, engineId, setEngineId, liveUrl, setLiveUrl, status, error, result, setResult, history, run };
}
