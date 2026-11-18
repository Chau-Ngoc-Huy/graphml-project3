import { useState, type ReactNode } from "react";
import { Command as CommandIcon, Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetTitle } from "@/components/ui/sheet";
import { useApp } from "@/state/AppState";
import { Sidebar } from "./Sidebar";
import { CommandMenu } from "./CommandMenu";
import { NAV } from "./nav";

export function AppShell({ children }: { children: ReactNode }) {
  const app = useApp();
  const [mobileOpen, setMobileOpen] = useState(false);
  const idx = NAV.findIndex((n) => n.id === app.route);
  const cur = NAV[idx];
  return (
    <div className="flex h-full">
      <aside className="hidden w-60 shrink-0 border-r bg-[hsl(222_14%_6%)] lg:block">
        <Sidebar />
      </aside>
      <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
        <SheetContent side="left" className="w-72 p-0">
          <SheetTitle className="sr-only">Điều hướng</SheetTitle>
          <SheetDescription className="sr-only">Các trang của Observatory</SheetDescription>
          <Sidebar onNavigate={() => setMobileOpen(false)} />
        </SheetContent>
      </Sheet>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-12 shrink-0 items-center gap-3 border-b px-3 md:px-6">
          <Button variant="ghost" size="icon" className="lg:hidden" onClick={() => setMobileOpen(true)} aria-label="Mở điều hướng"><Menu /></Button>
          <div className="flex min-w-0 items-center gap-2 text-[13px]">
            <span className="hidden text-muted-foreground sm:inline">Observatory</span>
            <span className="hidden text-muted-foreground/50 sm:inline">/</span>
            <span className="truncate font-medium">{cur?.label}</span>
            <span className="hidden font-mono text-[11px] text-muted-foreground md:inline">{idx + 1} / {NAV.length}</span>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <Button variant="outline" size="sm" className="gap-2 text-muted-foreground" onClick={() => app.setCommandOpen(true)}>
              <CommandIcon className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Tìm nhanh</span>
              <kbd className="hidden rounded border bg-muted px-1 font-mono text-[10px] md:inline">Ctrl K</kbd>
            </Button>
          </div>
        </header>
        <main id="main-scroll" className="min-h-0 flex-1 overflow-y-auto">
          {children}
          {idx >= 0 && idx < NAV.length - 1 && (
            <div className="mx-auto max-w-[1400px] px-4 pb-10 md:px-8">
              <button onClick={() => app.go(NAV[idx + 1].id)} className="group flex w-full items-center justify-between rounded-lg border bg-surface px-5 py-4 text-left transition-colors hover:border-primary/50">
                <div>
                  <div className="eyebrow">Tiếp theo</div>
                  <div className="mt-1 text-sm font-medium">{NAV[idx + 1].label}</div>
                  <div className="text-xs text-muted-foreground">{NAV[idx + 1].hint}</div>
                </div>
                <span className="font-mono text-xs text-muted-foreground transition-transform group-hover:translate-x-1">→</span>
              </button>
            </div>
          )}
        </main>
      </div>
      <CommandMenu />
    </div>
  );
}
