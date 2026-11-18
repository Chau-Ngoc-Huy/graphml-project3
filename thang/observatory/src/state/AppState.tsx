import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { BEST_ID, DEFAULT_COMPARE } from "@/lib/models";

export const ROUTES = ["overview", "journey", "playground", "compare", "graph", "results"] as const;
export type Route = (typeof ROUTES)[number];

interface AppState {
  route: Route;
  go: (r: Route) => void;
  /** Shared query context: the same input/selected test caption follows the user across pages. */
  input: string;
  setInput: (s: string) => void;
  queryIndex: number | null;
  setQueryIndex: (i: number | null) => void;
  playModels: string[];
  setPlayModels: (ids: string[]) => void;
  focusModel: string;
  setFocusModel: (id: string) => void;
  compareModels: string[];
  setCompareModels: (ids: string[]) => void;
  reference: string;
  setReference: (id: string) => void;
  commandOpen: boolean;
  setCommandOpen: (b: boolean) => void;
}

const Ctx = createContext<AppState | null>(null);

const readRoute = (): Route => {
  const h = window.location.hash.replace(/^#\/?/, "") as Route;
  return (ROUTES as readonly string[]).includes(h) ? h : "overview";
};

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [route, setRoute] = useState<Route>(readRoute);
  const [input, setInput] = useState("");
  const [queryIndex, setQueryIndex] = useState<number | null>(null);
  const [playModels, setPlayModels] = useState<string[]>(["clip", "clip_adamf", "t1_qa_ablation_no_edge", BEST_ID]);
  const [focusModel, setFocusModel] = useState<string>(BEST_ID);
  const [compareModels, setCompareModels] = useState<string[]>(DEFAULT_COMPARE);
  const [reference, setReference] = useState("clip");
  const [commandOpen, setCommandOpen] = useState(false);

  useEffect(() => {
    const on = () => setRoute(readRoute());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  const go = useCallback((r: Route) => {
    window.location.hash = `/${r}`;
    setRoute(r);
    document.getElementById("main-scroll")?.scrollTo({ top: 0 });
  }, []);

  const value = useMemo(
    () => ({ route, go, input, setInput, queryIndex, setQueryIndex, playModels, setPlayModels, focusModel, setFocusModel,
      compareModels, setCompareModels, reference, setReference, commandOpen, setCommandOpen }),
    [route, go, input, queryIndex, playModels, focusModel, compareModels, reference, commandOpen],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useApp() {
  const v = useContext(Ctx);
  if (!v) throw new Error("useApp outside AppStateProvider");
  return v;
}
