import { CartesianGrid, ComposedChart, ErrorBar, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS, COLORS, GRID, TOOLTIP_STYLE } from "./theme";

export interface CIDatum {
  name: string;
  delta: number;
  lo: number;
  hi: number;
}

/** Forest plot: paired-bootstrap Δ with 95 % CI against a zero line (right of zero = better). */
export function CIPlot({ data, height }: { data: CIDatum[]; height?: number }) {
  const rows = data.map((d) => ({ ...d, err: [d.delta - d.lo, d.hi - d.delta] as [number, number] }));
  const h = height ?? Math.max(160, rows.length * 30 + 40);
  return (
    <ResponsiveContainer width="100%" height={h}>
      <ComposedChart data={rows} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 4 }}>
        <CartesianGrid horizontal={false} {...GRID} />
        <XAxis type="number" dataKey="delta" {...AXIS} tickLine={false} domain={["auto", "auto"]} />
        <YAxis type="category" dataKey="name" width={170} {...AXIS} tickLine={false} axisLine={false} interval={0} />
        <ReferenceLine x={0} stroke="hsl(220 9% 60%)" />
        <Tooltip {...TOOLTIP_STYLE} formatter={(_v, _n, p) => [`${p.payload.delta.toFixed(2)}  [${p.payload.lo.toFixed(2)}, ${p.payload.hi.toFixed(2)}]`, "Δ MRR (CI95)"]} />
        <Scatter dataKey="delta" isAnimationActive={false} shape={(p: { cx?: number; cy?: number; payload?: CIDatum }) => (
          <circle cx={p.cx} cy={p.cy} r={4.5} fill={(p.payload?.lo ?? 0) > 0 ? COLORS.good : (p.payload?.hi ?? 0) < 0 ? COLORS.bad : COLORS.muted} stroke="hsl(222 14% 7%)" strokeWidth={1.5} />
        )}>
          <ErrorBar dataKey="err" direction="x" width={5} stroke="hsl(220 14% 75%)" strokeWidth={1.2} />
        </Scatter>
      </ComposedChart>
    </ResponsiveContainer>
  );
}
