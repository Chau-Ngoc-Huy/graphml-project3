import type { Family, ModelInfo } from "@/data/types";

export const FAMILY_LABEL: Record<Family, string> = {
  baseline: "Baseline",
  lexical: "Presence matching",
  kgc: "MMKG (AdaMF-MAT)",
  adapter: "Adapter",
  graph: "Graph (GNN)",
  ablation: "Ablation",
};

/** Stable categorical colour per family (validated palette order: blue, orange, aqua, yellow, magenta, violet). */
export const FAMILY_COLOR: Record<Family, string> = {
  baseline: "#9aa3ad",
  lexical: "#eda100",
  kgc: "#e87ba4",
  adapter: "#1baf7a",
  graph: "#6f7cf2",
  ablation: "#eb6834",
};

export const BASELINE_ID = "clip";
export const BEST_ID = "t1_qa_cn_no_sg_sage";
export const ADAPTER_ID = "t1_qa_ablation_no_edge";
export const SHUFFLE_ID = "t1_qa_cn_shuffle";

export const DEFAULT_COMPARE = ["clip", "clip_lex", "clip_adamf", ADAPTER_ID, BEST_ID, SHUFFLE_ID];

export function modelColor(m: ModelInfo) {
  return FAMILY_COLOR[m.family];
}
