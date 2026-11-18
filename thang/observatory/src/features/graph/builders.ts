import type { Edge, Node } from "@xyflow/react";
import { MarkerType } from "@xyflow/react";
import type { Dataset } from "@/data/types";
import { modelColor, FAMILY_LABEL } from "@/lib/models";
import type { GNodeData } from "./types";

type G = { nodes: Node<GNodeData>[]; edges: Edge[] };
const FLOW = "hsl(188 86% 53%)";
const MUTED = "hsl(222 10% 34%)";

const edge = (id: string, source: string, target: string, opts: Partial<Edge> & { path?: boolean; label?: string; dashed?: boolean } = {}): Edge => ({
  id, source, target,
  label: opts.label,
  labelStyle: { fill: "hsl(220 9% 62%)", fontSize: 10 },
  labelBgStyle: { fill: "hsl(222 14% 7%)" },
  animated: !!opts.path,
  style: { stroke: opts.path ? FLOW : MUTED, strokeWidth: opts.path ? 1.8 : 1, strokeDasharray: opts.dashed ? "4 3" : undefined },
  markerEnd: { type: MarkerType.ArrowClosed, color: opts.path ? FLOW : MUTED, width: 12, height: 12 },
  data: { path: !!opts.path },
});

/** Query execution graph for one test caption: query → concepts → objects (scene graph) → gold image → model ranks. */
export function buildQueryGraph(ds: Dataset, q: number, modelIds: string[], opts: { otherObjects: number; relations: boolean; models: boolean; path: boolean }): G {
  const img = ds.images[ds.queries.img[q]];
  const concepts = ds.queries.concepts[q].map((i) => ds.vocab[i]);
  const objs = new Set(img.objects);
  const matched = concepts.filter((c) => objs.has(c));
  const others = img.objects.filter((o) => !concepts.includes(o)).slice(0, opts.otherObjects);
  const shown = [...matched, ...others];
  const nodes: Node<GNodeData>[] = [];
  const edges: Edge[] = [];
  const P = opts.path;
  const gStr = ds.queries.groups[ds.queries.group[q]];
  nodes.push({ id: "q", type: "g", position: { x: 0, y: 0 }, data: { kind: "query", label: ds.queries.text[q].slice(0, 60) + (ds.queries.text[q].length > 60 ? "…" : ""), sub: `#${q} · ${gStr}`, onPath: P, payload: { text: ds.queries.text[q], index: q, group: gStr } } });
  const cy = (i: number, n: number) => (i - (n - 1) / 2) * 52;
  concepts.forEach((c, i) => {
    const inImg = objs.has(c);
    nodes.push({ id: `c:${c}`, type: "g", position: { x: 250, y: cy(i, concepts.length) }, data: { kind: "concept", label: c, sub: inImg ? "có trong ảnh" : "không có trong ảnh", tone: inImg ? "default" : "warn", onPath: P && inImg, payload: { concept: c, inImage: inImg } } });
    edges.push(edge(`e:q-${c}`, "q", `c:${c}`, { path: P && inImg, label: "liên kết" }));
  });
  shown.forEach((o, i) => {
    const isM = matched.includes(o);
    nodes.push({ id: `o:${o}`, type: "g", position: { x: 480, y: cy(i, shown.length) }, data: { kind: "object", label: o, sub: isM ? "khớp khái niệm" : "đối tượng khác", dim: !isM && P, onPath: P && isM, payload: { object: o } } });
    if (isM) edges.push(edge(`e:c-${o}`, `c:${o}`, `o:${o}`, { path: P, label: "khớp tên" }));
    edges.push(edge(`e:o-img-${o}`, `o:${o}`, "img", { path: P && isM, label: i === 0 ? "contains" : undefined }));
  });
  if (opts.relations) {
    const set = new Set(shown);
    img.relations.filter(([s, , o]) => set.has(s) && set.has(o)).slice(0, 14).forEach(([s, p, o], k) => {
      edges.push({ ...edge(`e:r${k}`, `o:${s}`, `o:${o}`, { dashed: true, label: p }), type: "smoothstep" });
    });
  }
  nodes.push({ id: "img", type: "g", position: { x: 720, y: 0 }, data: { kind: "image", label: `Ảnh VG ${img.vg}`, sub: `${img.objects.length} đối tượng · ${img.relations.length} quan hệ`, onPath: P, payload: { image: img } } });
  if (opts.models) {
    const live = modelIds.filter((id) => ds.ranks[id]);
    const best = Math.min(...live.map((id) => ds.ranks[id][q]));
    live.forEach((id, i) => {
      const m = ds.models.find((x) => x.id === id)!;
      const r = ds.ranks[id][q];
      nodes.push({ id: `m:${id}`, type: "g", position: { x: 960, y: cy(i, live.length) }, data: { kind: "model", label: m.short, sub: `hạng #${r} · ${FAMILY_LABEL[m.family]}`, color: modelColor(m), tone: r === 1 ? "good" : r > 10 ? "bad" : "default", onPath: P && r === best, payload: { modelId: id, rank: r } } });
      edges.push(edge(`e:img-${id}`, "img", `m:${id}`, { path: P && r === best, label: `#${r}` }));
    });
  }
  return { nodes, edges };
}

/** Khánh's 24 evaluation queries: query → subject / object entities; indirect ones connect only through a third object. */
export function buildKQueryGraph(ds: Dataset, group: "all" | "direct" | "indirect"): G {
  const qs = ds.results.kqueries.filter((q) => group === "all" || q.group === group);
  const nodes: Node<GNodeData>[] = [];
  const edges: Edge[] = [];
  const ents = new Map<string, number>();
  const vias = new Map<string, number>();
  qs.forEach((q) => { [q.triple[0], q.triple[2]].forEach((e) => ents.set(e, (ents.get(e) ?? 0) + 1)); if (q.group === "indirect") q.examples.forEach((x) => x.via && vias.set(x.via, (vias.get(x.via) ?? 0) + 1)); });
  qs.forEach((q, i) => {
    nodes.push({ id: `kq:${q.id}`, type: "g", position: { x: 0, y: i * 54 }, data: { kind: "query", label: q.text, sub: `${q.id} · ${q.group} · ${q.positives.length} ảnh đúng`, tone: q.group === "indirect" ? "warn" : "default", payload: { kq: q } } });
    edges.push(edge(`ks:${q.id}`, `kq:${q.id}`, `ent:${q.triple[0]}`, { label: "chủ thể" }));
    edges.push(edge(`ko:${q.id}`, `kq:${q.id}`, `ent:${q.triple[2]}`, { label: q.triple[1] }));
  });
  const entList = [...ents.entries()].sort((a, b) => b[1] - a[1]);
  const span = Math.max(qs.length * 54, entList.length * 46);
  entList.forEach(([e, n], i) => nodes.push({ id: `ent:${e}`, type: "g", position: { x: 380, y: (i * span) / Math.max(1, entList.length) }, data: { kind: "entity", label: e, sub: `${n} truy vấn`, payload: { entity: e, count: n } } }));
  const viaList = [...vias.entries()].sort((a, b) => b[1] - a[1]).slice(0, 18);
  viaList.forEach(([v, n], i) => nodes.push({ id: `via:${v}`, type: "g", position: { x: 720, y: (i * span) / Math.max(1, viaList.length) }, data: { kind: "via", label: v, sub: `nối ${n} lần`, payload: { via: v, count: n } } }));
  const keep = new Set(viaList.map(([v]) => v));
  qs.filter((q) => q.group === "indirect").forEach((q) => q.examples.forEach((x, k) => {
    if (x.via && keep.has(x.via)) edges.push(edge(`kv:${q.id}:${k}`, `ent:${q.triple[0]}`, `via:${x.via}`, { dashed: true }));
  }));
  return { nodes, edges };
}

/** Models × pipeline components they actually use (from each model's flags). */
export function buildModelGraph(ds: Dataset, owner: string): G {
  const comps = [
    { id: "clip", label: "CLIP đóng băng", sub: "cosine top-100", y: 0 },
    { id: "concepts", label: "Liên kết khái niệm", sub: "câu → nhãn", y: 70 },
    { id: "adapter", label: "Adapter học", sub: "chiếu 512×512", y: 140 },
    { id: "contains", label: "Cạnh ảnh–chứa–nhãn", sub: "336 581 cạnh", y: 210 },
    { id: "relations", label: "Quan hệ (scene graph / KG)", sub: "sg_rel / triple", y: 280 },
    { id: "conceptnet", label: "ConceptNet", sub: "32 826 cạnh", y: 350 },
    { id: "gnn", label: "GNN", sub: "GCN · GAT · SAGE", y: 420 },
    { id: "kgc", label: "MMKG completion", sub: "AdaMF-MAT", y: 490 },
  ];
  const nodes: Node<GNodeData>[] = comps.map((c) => ({ id: `cp:${c.id}`, type: "g", position: { x: 0, y: c.y }, data: { kind: "component", label: c.label, sub: c.sub, payload: { component: c.id } } }));
  const ms = ds.models.filter((m) => owner === "all" || m.owner.includes(owner));
  const edges: Edge[] = [];
  ms.forEach((m, i) => {
    nodes.push({ id: `m:${m.id}`, type: "g", position: { x: 420, y: i * 48 }, data: { kind: "model", label: m.short, sub: `${m.owner} · MRR ${m.official.full.MRR?.mean.toFixed(2)}`, color: modelColor(m), payload: { modelId: m.id } } });
    const uses = ["clip", "concepts"];
    if (m.flags.adapter) uses.push("adapter");
    if (m.flags.graph || m.family === "lexical") uses.push("contains");
    if (m.flags.relations && m.id !== "t1_qa_ablation_no_rel") uses.push("relations");
    if (m.flags.conceptnet) uses.push("conceptnet");
    if (m.flags.gnn) uses.push("gnn");
    if (m.family === "kgc") uses.push("kgc");
    uses.forEach((u) => edges.push(edge(`mu:${m.id}:${u}`, `cp:${u}`, `m:${m.id}`)));
  });
  return { nodes, edges };
}
