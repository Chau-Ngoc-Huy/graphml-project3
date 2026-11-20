import { lazy, Suspense } from "react";
import { AnimatePresence, motion } from "motion/react";
import { AlertTriangle } from "lucide-react";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Skeleton } from "@/components/ui/skeleton";
import { DataProvider, useDataState } from "@/data/DataProvider";
import { AppStateProvider, useApp, type Route } from "@/state/AppState";
import { AppShell } from "@/components/layout/AppShell";

const PAGES: Record<Route, React.LazyExoticComponent<() => JSX.Element>> = {
  overview: lazy(() => import("@/features/overview/ResearchOverview")),
  journey: lazy(() => import("@/features/pipeline/ResearchPipeline")),
  playground: lazy(() => import("@/features/playground/ModelPlayground")),
  compare: lazy(() => import("@/features/comparison/ModelComparison")),
  graph: lazy(() => import("@/features/graph/QueryGraphExplorer")),
  results: lazy(() => import("@/features/results/ExperimentResults")),
};

function PageSkeleton() {
  return (
    <div className="mx-auto max-w-[1400px] space-y-6 px-4 py-8 md:px-8">
      <Skeleton className="h-4 w-32" />
      <Skeleton className="h-8 w-2/3" />
      <div className="grid gap-4 md:grid-cols-3"><Skeleton className="h-40" /><Skeleton className="h-40" /><Skeleton className="h-40" /></div>
      <Skeleton className="h-72" />
    </div>
  );
}

function Routed() {
  const { route } = useApp();
  const Page = PAGES[route];
  return (
    <AnimatePresence mode="wait">
      <motion.div key={route} initial={{ opacity: 1, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }} transition={{ duration: 0.22, ease: [0.2, 0.7, 0.2, 1] }}>
        <Suspense fallback={<PageSkeleton />}>
          <Page />
        </Suspense>
      </motion.div>
    </AnimatePresence>
  );
}

function Gate() {
  const s = useDataState();
  if (s.status === "error")
    return (
      <div className="grid h-full place-items-center p-6">
        <div className="max-w-md space-y-2 rounded-lg border bg-surface p-6">
          <div className="flex items-center gap-2 font-medium text-bad"><AlertTriangle className="h-4 w-4" />Không tải được dữ liệu</div>
          <p className="text-sm text-muted-foreground">{s.error}</p>
          <p className="text-xs text-muted-foreground">Chạy <code className="font-mono">npm run data</code> để sinh public/data/*.json, rồi mở lại.</p>
        </div>
      </div>
    );
  if (s.status === "loading")
    return (
      <div className="grid h-full place-items-center">
        <div className="flex flex-col items-center gap-3 text-sm text-muted-foreground">
          <div className="h-1 w-48 overflow-hidden rounded bg-muted"><div className="h-full w-1/3 animate-[pulse_1s_ease-in-out_infinite] bg-primary" /></div>
          Đang nạp kết quả thực nghiệm…
        </div>
      </div>
    );
  return (
    <AppStateProvider>
      <AppShell>
        <Routed />
      </AppShell>
    </AppStateProvider>
  );
}

export default function App() {
  return (
    <TooltipProvider delayDuration={200}>
      <DataProvider>
        <Gate />
      </DataProvider>
    </TooltipProvider>
  );
}
