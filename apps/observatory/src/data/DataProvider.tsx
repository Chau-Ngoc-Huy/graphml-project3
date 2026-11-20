import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { loadDataset } from "./loader";
import type { Dataset, ModelInfo } from "./types";

interface DataState {
  status: "loading" | "ready" | "error";
  data?: Dataset;
  error?: string;
  modelById: Map<string, ModelInfo>;
}

const DataContext = createContext<DataState>({ status: "loading", modelById: new Map() });

export function DataProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<{ status: DataState["status"]; data?: Dataset; error?: string }>({ status: "loading" });
  useEffect(() => {
    let alive = true;
    loadDataset()
      .then((data) => alive && setState({ status: "ready", data }))
      .catch((e: Error) => alive && setState({ status: "error", error: e.message }));
    return () => {
      alive = false;
    };
  }, []);
  const value = useMemo<DataState>(
    () => ({ ...state, modelById: new Map((state.data?.models ?? []).map((m) => [m.id, m])) }),
    [state],
  );
  return <DataContext.Provider value={value}>{children}</DataContext.Provider>;
}

export function useDataState() {
  return useContext(DataContext);
}

/** Only call below the ready gate in App. */
export function useData(): Dataset & { modelById: Map<string, ModelInfo> } {
  const s = useContext(DataContext);
  if (!s.data) throw new Error("useData() called before data is ready");
  return { ...s.data, modelById: s.modelById };
}
