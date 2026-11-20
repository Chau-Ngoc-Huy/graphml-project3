import { useState, type ReactNode } from "react";
import { motion } from "motion/react";
import { ImageOff, Info, FileText } from "lucide-react";
import type { ModelInfo } from "@/data/types";
import { cn } from "@/lib/utils";
import { FAMILY_LABEL, modelColor } from "@/lib/models";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

export function PageHeader({ eyebrow, title, description, actions }: { eyebrow: string; title: ReactNode; description?: ReactNode; actions?: ReactNode }) {
  return (
    <header className="flex flex-col gap-4 border-b border-border/70 pb-6 md:flex-row md:items-end md:justify-between">
      <div className="max-w-3xl space-y-2">
        <div className="eyebrow">{eyebrow}</div>
        <h1 className="text-balance text-2xl font-semibold tracking-tight md:text-[28px]">{title}</h1>
        {description && <p className="max-w-[68ch] text-sm leading-relaxed text-muted-foreground">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}

export function Section({ title, description, aside, children, className }: { title: ReactNode; description?: ReactNode; aside?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={cn("space-y-4", className)}>
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="space-y-1">
          <h2 className="text-[15px] font-semibold tracking-tight">{title}</h2>
          {description && <p className="max-w-[72ch] text-[13px] leading-relaxed text-muted-foreground">{description}</p>}
        </div>
        {aside}
      </div>
      {children}
    </section>
  );
}

export function Stat({ label, value, sub, tone = "default", hint }: { label: string; value: ReactNode; sub?: ReactNode; tone?: "default" | "primary" | "flow" | "good" | "bad"; hint?: string }) {
  const toneCls = { default: "text-foreground", primary: "text-primary", flow: "text-flow", good: "text-good", bad: "text-bad" }[tone];
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <div className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-[0.12em] text-muted-foreground">
        {label}
        {hint && <InfoTip text={hint} />}
      </div>
      <div className={cn("font-mono text-2xl font-medium tabular tracking-tight", toneCls)}>{value}</div>
      {sub && <div className="text-xs text-muted-foreground">{sub}</div>}
    </div>
  );
}

export function InfoTip({ text }: { text: ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button type="button" aria-label="Explanation" className="inline-flex text-muted-foreground/70 hover:text-foreground">
          <Info className="h-3.5 w-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent className="normal-case tracking-normal">{text}</TooltipContent>
    </Tooltip>
  );
}

export function ModelDot({ model, size = 8 }: { model: ModelInfo; size?: number }) {
  return <span className="inline-block shrink-0 rounded-full" style={{ width: size, height: size, background: modelColor(model) }} />;
}

export function ModelChip({ model, active, onClick, className }: { model: ModelInfo; active?: boolean; onClick?: () => void; className?: string }) {
  const body = (
    <span className={cn("inline-flex min-w-0 items-center gap-1.5 rounded-md border px-2 py-1 text-xs transition-colors", active ? "border-primary/60 bg-primary/10 text-foreground" : "bg-surface text-muted-foreground hover:text-foreground", onClick && "cursor-pointer", className)}>
      <ModelDot model={model} />
      <span className="truncate">{model.short}</span>
    </span>
  );
  return (
    <Tooltip>
      <TooltipTrigger asChild>{onClick ? <button type="button" onClick={onClick}>{body}</button> : body}</TooltipTrigger>
      <TooltipContent>
        <div className="font-medium">{model.name}</div>
        <div className="text-muted-foreground">{FAMILY_LABEL[model.family]}</div>
        <div className="mt-1">{model.desc}</div>
      </TooltipContent>
    </Tooltip>
  );
}

export function RankPill({ rank, className }: { rank?: number; className?: string }) {
  if (rank == null) return <span className="text-xs text-muted-foreground">—</span>;
  const tone = rank === 1 ? "bg-good/15 text-good" : rank <= 10 ? "bg-flow/15 text-flow" : rank <= 100 ? "bg-warn/15 text-warn" : "bg-bad/15 text-bad";
  return <span className={cn("inline-flex min-w-[2.5rem] items-center justify-center rounded px-1.5 py-0.5 font-mono text-xs font-medium tabular", tone, className)}>#{rank}</span>;
}

export function Delta({ value, digits = 2, invert = false, suffix = "" }: { value: number; digits?: number; invert?: boolean; suffix?: string }) {
  const good = invert ? value < 0 : value > 0;
  const zero = Math.abs(value) < 1e-9;
  return (
    <span className={cn("font-mono text-xs tabular", zero ? "text-muted-foreground" : good ? "text-good" : "text-bad")}>
      {value > 0 ? "+" : value < 0 ? "−" : "±"}
      {Math.abs(value).toFixed(digits)}
      {suffix}
    </span>
  );
}

export function SourceNote({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-start gap-1.5 text-[11px] leading-relaxed text-muted-foreground/80">
      <FileText className="mt-px h-3 w-3 shrink-0" />
      <span className="font-mono">{children}</span>
    </div>
  );
}

export function ImageThumb({ url, alt, className, badge }: { url?: string; alt: string; className?: string; badge?: ReactNode }) {
  const [err, setErr] = useState(false);
  return (
    <div className={cn("relative overflow-hidden rounded-md border bg-muted", className)}>
      {url && !err ? (
        <img src={url} alt={alt} loading="lazy" onError={() => setErr(true)} className="h-full w-full object-cover" />
      ) : (
        <div className="flex h-full w-full items-center justify-center text-muted-foreground"><ImageOff className="h-5 w-5" /></div>
      )}
      {badge && <div className="absolute left-1.5 top-1.5">{badge}</div>}
    </div>
  );
}

export function Panel({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn("panel", className)}>{children}</div>;
}

export function FadeIn({ children, delay = 0, className }: { children: ReactNode; delay?: number; className?: string }) {
  return (
    <motion.div initial={{ opacity: 1, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.28, delay, ease: [0.2, 0.7, 0.2, 1] }} className={className}>
      {children}
    </motion.div>
  );
}

export function EmptyState({ icon, title, children }: { icon?: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed px-6 py-10 text-center">
      {icon && <div className="text-muted-foreground">{icon}</div>}
      <div className="text-sm font-medium">{title}</div>
      {children && <div className="max-w-md text-xs leading-relaxed text-muted-foreground">{children}</div>}
    </div>
  );
}
