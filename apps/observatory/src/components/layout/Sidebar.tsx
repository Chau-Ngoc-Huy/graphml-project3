import { motion } from "motion/react";
import { Database } from "lucide-react";
import { useApp } from "@/state/AppState";
import { useData } from "@/data/DataProvider";
import { cn } from "@/lib/utils";
import { NAV } from "./nav";

export function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const { route, go } = useApp();
  const { meta, models } = useData();
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2.5 px-4 pb-5 pt-5">
        <div className="relative grid h-8 w-8 place-items-center rounded-md border bg-surface-raised">
          <svg viewBox="0 0 24 24" className="h-4.5 w-4.5" fill="none" stroke="currentColor" strokeWidth={1.6}>
            <circle cx="6" cy="7" r="2.2" className="text-primary" stroke="hsl(var(--primary))" />
            <circle cx="18" cy="6" r="2.2" stroke="hsl(var(--flow))" />
            <circle cx="12" cy="17.5" r="2.2" stroke="hsl(var(--primary))" />
            <path d="M7.9 8.3 10.8 15.6M16.6 7.8 13.2 15.6M8.2 6.8 15.8 6.2" stroke="hsl(var(--muted-foreground))" />
          </svg>
        </div>
        <div className="min-w-0 leading-tight">
          <div className="text-sm font-semibold tracking-tight">MMXAI Observatory</div>
          <div className="truncate text-[11px] text-muted-foreground">Project 3 · Graph ML</div>
        </div>
      </div>
      <nav className="flex-1 space-y-0.5 px-2">
        {NAV.map((n, i) => {
          const active = route === n.id;
          return (
            <button
              key={n.id}
              onClick={() => { go(n.id); onNavigate?.(); }}
              className={cn("group relative flex w-full items-center gap-3 rounded-md px-2.5 py-2 text-left text-[13px] transition-colors", active ? "text-foreground" : "text-muted-foreground hover:bg-accent/50 hover:text-foreground")}
            >
              {active && <motion.span layoutId="nav-active" className="absolute inset-0 rounded-md border border-border bg-surface-raised" transition={{ type: "spring", stiffness: 500, damping: 40 }} />}
              <span className="relative font-mono text-[10px] text-muted-foreground/70">{String(i + 1).padStart(2, "0")}</span>
              <n.icon className={cn("relative h-4 w-4", active && "text-primary")} />
              <span className="relative flex-1 truncate">{n.label}</span>
            </button>
          );
        })}
      </nav>
      <div className="m-3 space-y-2 rounded-md border bg-surface p-3 text-[11px] leading-relaxed text-muted-foreground">
        <div className="flex items-center gap-1.5 text-foreground"><Database className="h-3.5 w-3.5 text-flow" />Dữ liệu thật của nhóm</div>
        <div className="font-mono tabular">{models.length} cấu hình · {meta.dataset.test_queries.toLocaleString("vi-VN")} query test</div>
        <div>{meta.checks.length} kiểm tra chéo khớp số đã báo cáo</div>
      </div>
    </div>
  );
}
