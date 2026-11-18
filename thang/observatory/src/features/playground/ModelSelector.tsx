import { Check, Layers } from "lucide-react";
import { useData } from "@/data/DataProvider";
import type { Family } from "@/data/types";
import { FAMILY_LABEL } from "@/lib/models";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { ModelChip, ModelDot } from "@/components/common";

const ORDER: Family[] = ["baseline", "lexical", "kgc", "adapter", "graph", "ablation"];

export function ModelSelector({ selected, onChange, focus, onFocus }: { selected: string[]; onChange: (ids: string[]) => void; focus?: string; onFocus?: (id: string) => void }) {
  const ds = useData();
  const toggle = (id: string) => onChange(selected.includes(id) ? selected.filter((x) => x !== id) : [...selected, id]);
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {selected.map((id) => {
        const m = ds.modelById.get(id);
        return m ? <ModelChip key={id} model={m} active={focus === id} onClick={onFocus ? () => onFocus(id) : undefined} /> : null;
      })}
      <Popover>
        <PopoverTrigger asChild>
          <Button variant="ghost" size="sm" className="h-7 gap-1.5 text-muted-foreground"><Layers className="h-3.5 w-3.5" />{selected.length} / {ds.models.length} mô hình</Button>
        </PopoverTrigger>
        <PopoverContent className="w-[340px] p-1">
          <div className="max-h-[420px] overflow-y-auto">
            {ORDER.map((f) => {
              const ms = ds.models.filter((m) => m.family === f);
              if (!ms.length) return null;
              return (
                <div key={f} className="py-1">
                  <div className="px-2 py-1 text-[10px] uppercase tracking-wider text-muted-foreground">{FAMILY_LABEL[f]}</div>
                  {ms.map((m) => {
                    const on = selected.includes(m.id);
                    return (
                      <button key={m.id} onClick={() => toggle(m.id)} className={cn("flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-[13px] hover:bg-accent", !m.hasRanks && "opacity-70")}>
                        <span className={cn("grid h-4 w-4 place-items-center rounded-sm border", on ? "border-primary bg-primary text-primary-foreground" : "border-muted-foreground/40")}>{on && <Check className="h-3 w-3" strokeWidth={3} />}</span>
                        <ModelDot model={m} />
                        <span className="flex-1 truncate">{m.name}</span>
                        <span className="text-[10px] text-muted-foreground">{m.hasRanks ? m.owner : "chỉ số 3 seed"}</span>
                      </button>
                    );
                  })}
                </div>
              );
            })}
          </div>
          <div className="flex justify-between border-t p-1.5">
            <Button variant="ghost" size="sm" onClick={() => onChange(ds.models.filter((m) => m.hasRanks).map((m) => m.id))}>Chọn tất cả</Button>
            <Button variant="ghost" size="sm" onClick={() => onChange(["clip"])}>Chỉ CLIP</Button>
          </div>
        </PopoverContent>
      </Popover>
    </div>
  );
}
