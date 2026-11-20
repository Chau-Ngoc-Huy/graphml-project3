export type GKind = "query" | "concept" | "object" | "image" | "model" | "entity" | "via" | "component";

export type GNodeData = {
  kind: GKind;
  label: string;
  sub?: string;
  tone?: "default" | "good" | "bad" | "warn" | "muted";
  color?: string;
  dim?: boolean;
  onPath?: boolean;
  payload?: Record<string, unknown>;
};

export const KIND_STYLE: Record<GKind, { label: string; cls: string; dot: string }> = {
  query: { label: "Query", cls: "border-primary/70 bg-primary/10", dot: "bg-primary" },
  concept: { label: "Concept", cls: "border-flow/60 bg-flow/10", dot: "bg-flow" },
  object: { label: "Object trong ảnh", cls: "border-border bg-surface-raised", dot: "bg-muted-foreground" },
  image: { label: "Gold image", cls: "border-good/60 bg-good/10", dot: "bg-good" },
  model: { label: "Model", cls: "border-border bg-surface", dot: "bg-primary" },
  entity: { label: "Entity", cls: "border-flow/60 bg-flow/10", dot: "bg-flow" },
  via: { label: "Via object", cls: "border-warn/50 bg-warn/10", dot: "bg-warn" },
  component: { label: "Thành phần pipeline", cls: "border-border bg-surface-raised", dot: "bg-flow" },
};
