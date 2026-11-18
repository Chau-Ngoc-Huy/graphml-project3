import { useMemo, useState } from "react";
import { CornerDownLeft, Loader2, Play, Shuffle } from "lucide-react";
import { useData } from "@/data/DataProvider";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Badge } from "@/components/ui/badge";

/** Query box with caption autocomplete over the 10 696 test captions (substring match). */
export function QueryInput({ value, onChange, onRun, running }: { value: string; onChange: (s: string) => void; onRun: (forceIndex?: number) => void; running: boolean }) {
  const ds = useData();
  const [open, setOpen] = useState(false);
  const matches = useMemo(() => {
    const q = value.trim().toLowerCase();
    if (q.length < 3) return [];
    const out: number[] = [];
    for (let i = 0; i < ds.queries.text.length && out.length < 8; i++) if (ds.queries.text[i].toLowerCase().includes(q)) out.push(i);
    return out;
  }, [value, ds]);

  const random = () => {
    const i = Math.floor(Math.random() * ds.queries.text.length);
    onChange(ds.queries.text[i]);
    onRun(i);
  };

  return (
    <div className="flex flex-col gap-2 sm:flex-row">
      <Popover open={open && matches.length > 0} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <div className="relative flex-1">
            <textarea
              value={value}
              rows={2}
              onFocus={() => setOpen(true)}
              onChange={(e) => { onChange(e.target.value); setOpen(true); }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  setOpen(false);
                  onRun();
                }
              }}
              placeholder="Nhập một câu mô tả ảnh (tiếng Anh), ví dụ: a man riding a surfboard on a wave"
              className="w-full resize-none rounded-md border bg-background px-3.5 py-3 pr-24 font-mono text-[13px] leading-relaxed placeholder:font-sans placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            />
            <span className="pointer-events-none absolute bottom-3 right-3 hidden items-center gap-1 text-[10px] text-muted-foreground sm:flex"><CornerDownLeft className="h-3 w-3" />Enter để chạy</span>
          </div>
        </PopoverTrigger>
        <PopoverContent className="w-[min(640px,90vw)] p-1" onOpenAutoFocus={(e) => e.preventDefault()}>
          <div className="px-2 py-1.5 text-[11px] uppercase tracking-wider text-muted-foreground">Caption test khớp — có kết quả thật của mọi mô hình</div>
          {matches.map((i) => (
            <button key={i} onClick={() => { onChange(ds.queries.text[i]); setOpen(false); onRun(i); }} className="flex w-full items-start gap-2 rounded px-2 py-1.5 text-left text-[13px] hover:bg-accent">
              <Badge variant="outline" className="mt-0.5 shrink-0 font-mono">#{i}</Badge>
              <span className="line-clamp-2">{ds.queries.text[i]}</span>
            </button>
          ))}
        </PopoverContent>
      </Popover>
      <div className="flex gap-2 sm:flex-col">
        <Button onClick={() => onRun()} disabled={running || !value.trim()} className="flex-1 sm:w-32">
          {running ? <Loader2 className="animate-spin" /> : <Play />} Chạy
        </Button>
        <Button variant="outline" onClick={random} className="flex-1 sm:w-32" title="Lấy ngẫu nhiên một caption test">
          <Shuffle /> Ngẫu nhiên
        </Button>
      </div>
    </div>
  );
}
