import { AlertCircle, Clock, Cpu, Info } from "lucide-react";
import { motion } from "motion/react";
import { useData } from "@/data/DataProvider";
import type { ModelRun } from "@/execution/types";
import { modelColor, FAMILY_LABEL } from "@/lib/models";
import { fmt, fmtDuration } from "@/lib/format";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { ImageThumb, ModelDot, RankPill } from "@/components/common";
import { RankMeter } from "./RankMeter";

export function ModelOutputPanel({ run, focused, onFocus, index = 0 }: { run: ModelRun; focused?: boolean; onFocus?: () => void; index?: number }) {
  const ds = useData();
  const m = ds.modelById.get(run.modelId);
  const pool = ds.images.length;
  const cost = m?.official.cost;
  const label = m?.name ?? run.modelId;
  const delta = run.goldRank != null && run.baselineRank != null ? run.baselineRank - run.goldRank : null;
  return (
    <motion.div initial={{ opacity: 1, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.22, delay: index * 0.03 }}
      onClick={onFocus}
      className={cn("flex min-w-0 flex-col gap-3 rounded-lg border bg-surface p-4 transition-colors", onFocus && "cursor-pointer hover:border-muted-foreground/40", focused && "border-primary/70 bg-primary/[0.04]")}>
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            {m && <ModelDot model={m} />}
            <span className="truncate text-[13px] font-medium">{label}</span>
          </div>
          {m && <div className="mt-0.5 text-[11px] text-muted-foreground">{FAMILY_LABEL[m.family]}</div>}
        </div>
        {run.status === "success" ? <RankPill rank={run.goldRank} /> : <Badge variant={run.status === "error" ? "bad" : "muted"}>{run.status === "error" ? "lỗi" : "không khả dụng"}</Badge>}
      </div>

      {run.status === "success" && run.goldRank != null && m && (
        <>
          <RankMeter rank={run.goldRank} base={run.baselineRank} pool={pool} color={modelColor(m)} />
          <div className="mt-3 grid grid-cols-3 gap-2 text-[11px]">
            <div><div className="text-muted-foreground">So với CLIP</div>
              <div className={cn("font-mono tabular", delta == null || delta === 0 ? "text-muted-foreground" : delta > 0 ? "text-good" : "text-bad")}>
                {delta == null ? "—" : delta === 0 ? "giữ nguyên" : `${delta > 0 ? "↑" : "↓"} ${Math.abs(delta)} rank`}
              </div></div>
            <div><div className="text-muted-foreground">Trúng</div>
              <div className="flex gap-1 font-mono">{[1, 5, 10].map((k) => <span key={k} className={run.goldRank! <= k ? "text-good" : "text-muted-foreground/50"}>@{k}</span>)}</div></div>
            <div><div className="text-muted-foreground">1 / rank</div><div className="font-mono tabular">{fmt(1 / run.goldRank, 3)}</div></div>
          </div>
        </>
      )}
      {run.status === "success" && run.retrieved && run.retrieved.length > 0 && (
        <div className="grid grid-cols-5 gap-1.5">
          {run.retrieved.slice(0, 5).map((r) => <ImageThumb key={r.rank} url={r.url} alt={r.label} className={cn("aspect-square", r.isGold && "ring-2 ring-good")} />)}
        </div>
      )}
      {run.note && (
        <div className="flex gap-2 rounded-md bg-muted/60 p-2.5 text-[11px] leading-relaxed text-muted-foreground">
          {run.status === "error" ? <AlertCircle className="mt-px h-3.5 w-3.5 shrink-0 text-bad" /> : <Info className="mt-px h-3.5 w-3.5 shrink-0" />}
          {run.note}
        </div>
      )}
      <div className="mt-auto flex flex-wrap gap-x-4 gap-y-1 border-t border-border/60 pt-2.5 text-[10px] text-muted-foreground">
        <span className="flex items-center gap-1" title="Thời gian engine trả kết quả, đo trong trình duyệt"><Clock className="h-3 w-3" />{run.latencyMs < 1 ? "< 1" : fmt(run.latencyMs, 1)} ms</span>
        {cost?.sec_per_sample != null && <span className="flex items-center gap-1" title={`Ghi nhận khi chạy thật: ${cost.device}`}><Cpu className="h-3 w-3" />{fmt(cost.sec_per_sample * 1000, 0)} ms/query (T4)</span>}
        {cost?.wall_time_s != null && <span title={cost.scope}>huấn luyện + đánh giá {fmtDuration(cost.wall_time_s)}</span>}
      </div>
    </motion.div>
  );
}
