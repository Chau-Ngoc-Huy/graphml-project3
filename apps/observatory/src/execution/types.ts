import type { LinkResult } from "@/lib/linker";

export type RunStatus = "success" | "unavailable" | "error";

export interface RetrievedImage {
  imageIndex?: number; // index into dataset.images (replay engine)
  url?: string;
  label: string;
  score?: number;
  rank: number;
  isGold?: boolean;
  matched?: string[];
  detail?: Record<string, number | string | null>;
}

export interface ModelRun {
  modelId: string;
  status: RunStatus;
  /** Wall-clock time this engine spent producing the output, measured in the browser. */
  latencyMs: number;
  /** Rank of the ground-truth image in the full test pool (2 138 images). */
  goldRank?: number;
  baselineRank?: number;
  retrieved?: RetrievedImage[];
  note?: string;
  raw?: unknown;
}

export type Resolution =
  | { mode: "test-caption"; queryIndex: number; similarity: number }
  | { mode: "free-text" };

export interface ExecutionResult {
  id: string;
  engine: string;
  input: string;
  resolution: Resolution;
  linking: LinkResult;
  goldImageIndex?: number;
  /** Browser-side concept matching (the non-learned part of the pipeline) — runs for ANY text. */
  conceptMatch: { latencyMs: number; retrieved: RetrievedImage[] };
  runs: ModelRun[];
  suggestions: { queryIndex: number; similarity: number }[];
  totalMs: number;
  finishedAt: number;
}

export interface ExecutionEngine {
  id: string;
  label: string;
  description: string;
  execute(input: string, modelIds: string[], opts?: { forceQueryIndex?: number }): Promise<ExecutionResult>;
}
