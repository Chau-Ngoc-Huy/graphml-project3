import { memo } from "react";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { cn } from "@/lib/utils";
import { KIND_STYLE, type GNodeData } from "./types";

function GraphNodeImpl({ data, selected }: NodeProps<Node<GNodeData>>) {
  const s = KIND_STYLE[data.kind];
  return (
    <div
      className={cn(
        "max-w-[210px] rounded-md border px-2.5 py-1.5 text-left shadow-sm transition-all",
        s.cls,
        data.dim && "opacity-35",
        data.onPath && "ring-2 ring-flow/70",
        selected && "ring-2 ring-primary",
        data.tone === "bad" && "border-bad/60",
        data.tone === "warn" && "border-dashed border-warn/70",
      )}
    >
      <Handle type="target" position={Position.Left} className="!h-1.5 !w-1.5 !border-0 !bg-muted-foreground/60" />
      <div className="flex items-center gap-1.5">
        <span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", s.dot)} style={data.color ? { background: data.color } : undefined} />
        <span className="truncate text-[12px] font-medium leading-tight">{data.label}</span>
      </div>
      {data.sub && <div className="mt-0.5 truncate font-mono text-[10px] text-muted-foreground">{data.sub}</div>}
      <Handle type="source" position={Position.Right} className="!h-1.5 !w-1.5 !border-0 !bg-muted-foreground/60" />
    </div>
  );
}

export const GraphNode = memo(GraphNodeImpl);
