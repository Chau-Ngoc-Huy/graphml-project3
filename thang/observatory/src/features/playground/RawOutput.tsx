import { lazy, Suspense } from "react";
import { Skeleton } from "@/components/ui/skeleton";

const Editor = lazy(() => import("@monaco-editor/react"));

/** Read-only Monaco viewer for raw engine output. Monaco loads from jsDelivr on first use. */
export function RawOutput({ value, height = 420 }: { value: unknown; height?: number }) {
  return (
    <div className="overflow-hidden rounded-md border">
      <Suspense fallback={<Skeleton className="w-full" style={{ height }} />}>
        <Editor
          height={height}
          defaultLanguage="json"
          value={JSON.stringify(value, null, 2)}
          theme="vs-dark"
          loading={<Skeleton className="w-full" style={{ height }} />}
          options={{ readOnly: true, minimap: { enabled: false }, fontSize: 12, fontFamily: "JetBrains Mono Variable, monospace", scrollBeyondLastLine: false, lineNumbers: "off", folding: true, wordWrap: "on", renderLineHighlight: "none" }}
        />
      </Suspense>
    </div>
  );
}
