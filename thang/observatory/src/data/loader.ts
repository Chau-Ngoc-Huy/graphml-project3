import type { Dataset } from "./types";

const FILES = ["meta", "models", "results", "queries", "images", "vocab", "ranks"] as const;

/** Loads the static data layer from public/data. Swap this for an API client without touching the UI. */
export async function loadDataset(base = import.meta.env.BASE_URL): Promise<Dataset> {
  const entries = await Promise.all(
    FILES.map(async (name) => {
      const res = await fetch(`${base}data/${name}.json`);
      if (!res.ok) throw new Error(`Không tải được data/${name}.json (${res.status})`);
      return [name, await res.json()] as const;
    }),
  );
  return Object.fromEntries(entries) as unknown as Dataset;
}
