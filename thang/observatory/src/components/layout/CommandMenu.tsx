import { useEffect, useMemo } from "react";
import { FlaskConical, Sparkles } from "lucide-react";
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { useApp } from "@/state/AppState";
import { useData } from "@/data/DataProvider";
import { buildExamples } from "@/lib/examples";
import { ModelDot } from "@/components/common";
import { NAV } from "./nav";

export function CommandMenu() {
  const app = useApp();
  const ds = useData();
  const examples = useMemo(() => buildExamples(ds), [ds]);
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        app.setCommandOpen(!app.commandOpen);
      }
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [app]);
  const close = () => app.setCommandOpen(false);
  return (
    <Dialog open={app.commandOpen} onOpenChange={app.setCommandOpen}>
      <DialogContent className="p-0">
        <DialogTitle className="sr-only">Tìm nhanh</DialogTitle>
        <DialogDescription className="sr-only">Chuyển trang, mở ví dụ hoặc chọn mô hình</DialogDescription>
        <Command>
          <CommandInput placeholder="Đi tới trang, ví dụ truy vấn, mô hình…" />
          <CommandList>
            <CommandEmpty>Không có kết quả.</CommandEmpty>
            <CommandGroup heading="Trang">
              {NAV.map((n) => (
                <CommandItem key={n.id} value={`trang ${n.label} ${n.hint}`} onSelect={() => { app.go(n.id); close(); }}>
                  <n.icon /> {n.label}<span className="ml-auto text-xs text-muted-foreground">{n.hint}</span>
                </CommandItem>
              ))}
            </CommandGroup>
            <CommandGroup heading="Ví dụ truy vấn">
              {examples.map((e) => (
                <CommandItem key={e.id} value={`vi du ${e.label} ${e.text}`} onSelect={() => { app.setInput(e.text); app.setQueryIndex(e.queryIndex ?? null); app.go("playground"); close(); }}>
                  <Sparkles className="text-flow" />
                  <span className="truncate">{e.text}</span>
                  <span className="ml-auto shrink-0 text-xs text-muted-foreground">{e.label}</span>
                </CommandItem>
              ))}
            </CommandGroup>
            <CommandGroup heading="Mô hình">
              {ds.models.map((m) => (
                <CommandItem key={m.id} value={`mo hinh ${m.name} ${m.short} ${m.owner}`} onSelect={() => { app.setFocusModel(m.id); if (!app.playModels.includes(m.id)) app.setPlayModels([...app.playModels, m.id]); app.go("playground"); close(); }}>
                  <ModelDot model={m} /> {m.name}<span className="ml-auto text-xs text-muted-foreground">{m.owner}</span>
                  <FlaskConical className="text-muted-foreground" />
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </DialogContent>
    </Dialog>
  );
}
