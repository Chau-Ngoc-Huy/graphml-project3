import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, LabelList, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS, GRID, TOOLTIP_STYLE } from "./theme";

export interface MetricBarDatum {
  name: string;
  value: number;
  std?: number | null;
  color: string;
  highlight?: boolean;
}

/** Horizontal bars with optional ± std error bars and a reference line (e.g. CLIP). */
export function MetricBars({ data, domain, reference, referenceLabel, unit = "", height }: { data: MetricBarDatum[]; domain?: [number, number]; reference?: number; referenceLabel?: string; unit?: string; height?: number }) {
  const rows = data.map((d) => ({ ...d, err: d.std ?? 0 }));
  const h = height ?? Math.max(160, rows.length * 30 + 40);
  return (
    <ResponsiveContainer width="100%" height={h}>
      <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 56, bottom: 4, left: 4 }} barCategoryGap={6}>
        <CartesianGrid horizontal={false} {...GRID} />
        <XAxis type="number" domain={domain ?? ["auto", "auto"]} {...AXIS} tickLine={false} />
        <YAxis type="category" dataKey="name" width={170} {...AXIS} tickLine={false} axisLine={false} interval={0} />
        <Tooltip {...TOOLTIP_STYLE} formatter={(v: number, _n, p) => [`${v.toFixed(2)}${unit}${p.payload.std ? ` ± ${p.payload.std.toFixed(2)}` : ""}`, "giá trị"]} />
        {reference != null && <ReferenceLine x={reference} stroke="hsl(220 9% 60%)" strokeDasharray="4 3" label={{ value: referenceLabel, position: "top", fill: "hsl(220 9% 60%)", fontSize: 10 }} />}
        <Bar dataKey="value" radius={[0, 3, 3, 0]} isAnimationActive={false}>
          {rows.map((d) => <Cell key={d.name} fill={d.color} fillOpacity={d.highlight === false ? 0.45 : 1} />)}
          <ErrorBar dataKey="err" width={4} stroke="hsl(220 14% 85%)" strokeWidth={1} direction="x" />
          <LabelList dataKey="value" position="right" offset={12} formatter={(v: number) => v.toFixed(2)} style={{ fill: "hsl(220 14% 80%)", fontSize: 11, fontFamily: "JetBrains Mono Variable" }} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
