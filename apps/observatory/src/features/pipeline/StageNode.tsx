import { memo } from "react";
import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";
import { cn } from "@/lib/utils";

export type StageNodeData = { index: number; title: string; sub: string; status: "done" | "partial" | "planned"; selected: boolean; branch?: boolean };

const dot = { done: "bg-good", partial: "bg-warn", planned: "bg-muted-foreground" };

function StageNodeImpl({ data }: NodeProps<Node<StageNodeData>>) {
  return (
    <div
      className={cn(
        "w-[168px] cursor-pointer rounded-md border bg-surface px-3 py-2.5 shadow-sm transition-all",
        data.selected ? "border-primary ring-2 ring-primary/30" : "hover:border-muted-foreground/40",
        data.branch && "border-dashed",
      )}
    >
      <Handle type="target" position={Position.Left} className="!h-1.5 !w-1.5 !border-0 !bg-flow" />
      <div className="flex items-center gap-2">
        <span className="font-mono text-[10px] text-muted-foreground">{data.branch ? "↳" : String(data.index).padStart(2, "0")}</span>
        <span className={cn("h-1.5 w-1.5 rounded-full", dot[data.status])} />
      </div>
      <div className="mt-1 text-[13px] font-medium leading-snug">{data.title}</div>
      <div className="mt-0.5 truncate text-[11px] text-muted-foreground">{data.sub}</div>
      <Handle type="source" position={Position.Right} className="!h-1.5 !w-1.5 !border-0 !bg-flow" />
    </div>
  );
}

export const StageNode = memo(StageNodeImpl);
