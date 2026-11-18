import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS, GRID, TOOLTIP_STYLE } from "./theme";

export interface Series {
  key: string;
  label: string;
  color: string;
}

export function GroupedBars({ data, series, xKey, domain, height = 260, unit = "" }: { data: Record<string, number | string>[]; series: Series[]; xKey: string; domain?: [number | "auto", number | "auto"]; height?: number; unit?: string }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -8 }} barGap={2} barCategoryGap="22%">
        <CartesianGrid vertical={false} {...GRID} />
        <XAxis dataKey={xKey} {...AXIS} tickLine={false} />
        <YAxis domain={domain ?? [0, "auto"]} {...AXIS} tickLine={false} axisLine={false} />
        <Tooltip {...TOOLTIP_STYLE} formatter={(v: number, name) => [`${v.toFixed(2)}${unit}`, name]} />
        <Legend iconType="square" iconSize={8} wrapperStyle={{ fontSize: 11, color: "hsl(220 9% 60%)", paddingTop: 6 }} />
        {series.map((s) => <Bar key={s.key} dataKey={s.key} name={s.label} fill={s.color} radius={[3, 3, 0, 0]} maxBarSize={28} isAnimationActive={false} />)}
      </BarChart>
    </ResponsiveContainer>
  );
}
