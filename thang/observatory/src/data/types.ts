/** Typed shapes of the files produced by scripts/build_data.py (all from REAL team results). */
export const METRIC_KEYS = ["R@1", "R@5", "R@10", "MRR", "nDCG@10"] as const;
export type MetricKey = (typeof METRIC_KEYS)[number];

export interface Stat {
  mean: number;
  std: number | null;
}
export type MetricSet = Partial<Record<MetricKey, Stat>>;

export type Family = "baseline" | "lexical" | "kgc" | "adapter" | "graph" | "ablation";
export type QueryGroup = "direct" | "indirect" | "no_concept";

export interface ModelFlags {
  adapter: boolean;
  graph: boolean;
  relations: boolean;
  conceptnet: boolean;
  learned: boolean;
  gnn: string | null;
}

export interface ModelCost {
  wall_time_s?: number | null;
  peak_vram_gb?: number | null;
  sec_per_sample?: number | null;
  device?: string;
  scope?: string;
}

export interface GraphStats {
  n_label?: number;
  E_contains?: number;
  E_sg_rel?: number;
  E_concept?: number;
  variant?: string;
}

export interface ModelOfficial {
  protocol: string;
  test1k: MetricSet;
  full: MetricSet;
  groups1k: Partial<Record<QueryGroup, MetricSet>>;
  alpha?: number | null;
  cost?: ModelCost;
  graph?: GraphStats | null;
  config?: { gnn?: string | null; variant?: string; extra?: string | null };
  source: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  short: string;
  owner: string;
  family: Family;
  desc: string;
  flags: ModelFlags;
  official: ModelOfficial;
  hasRanks: boolean;
  seed0FullMRR?: number;
}

export interface Meta {
  title: string;
  dataset: {
    name: string;
    pool: Record<"train" | "val" | "test", number>;
    split: Record<"train" | "val" | "test", number>;
    images_encoded: number;
    captions: number;
    clip: string;
    test_queries: number;
    test_images: number;
    vocab: number;
  };
  protocol: { gallery_size: number; top_k: number; seeds: number[]; seed_data: number };
  checks: { model: string; mrr_from_ranks?: number; reported?: number; check?: string; ok?: boolean }[];
  sources: Record<string, string>;
}

export interface QueriesData {
  text: string[];
  img: number[];
  group: number[];
  concepts: number[][];
  groups: QueryGroup[];
}

export interface ImageInfo {
  vg: number;
  coco: number;
  url: string;
  objects: string[];
  relations: [string, string, string][];
}

export interface PairedDelta {
  delta: number;
  ci95: [number, number];
  p_value: number;
}
export interface PairedRow {
  MRR: Record<"all" | "direct" | "indirect", number>;
  vs_AdaMF?: Record<"all" | "direct" | "indirect", PairedDelta>;
}
export interface Level2Row {
  group: string;
  method: string;
  "R@1": number;
  "R@5": number;
  "R@10": number;
  MRR: number;
  "nDCG@10": number;
}
export interface ReproRow {
  dataset: string;
  who: "paper" | "ours";
  MRR: number;
  "Hits@1": number;
  "Hits@3": number;
  "Hits@10": number;
}
export interface KQuery {
  id: string;
  group: "direct" | "indirect";
  text: string;
  triple: [string, string, string];
  positives: number[];
  examples: { image: number; via?: string }[];
}
export interface ResultsData {
  paired: Record<string, PairedRow>;
  level2: Level2Row[];
  repro: ReproRow[];
  i2t: { test1k: MetricSet; full: MetricSet };
  kqueries: KQuery[];
}

export type RanksData = Record<string, number[]>;

export interface Dataset {
  meta: Meta;
  models: ModelInfo[];
  results: ResultsData;
  queries: QueriesData;
  images: ImageInfo[];
  vocab: string[];
  ranks: RanksData;
}
