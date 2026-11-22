import { useMemo, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Download, TriangleAlert, TrendingUp } from "lucide-react";

import { chartColor } from "@/components/charts";
import {
  formatValue,
  prettyName,
  shortLabel,
  type Dataset,
} from "@/lib/dataset";
import {
  forecastColumn,
  forecastToCsv,
  type ForecastResult,
} from "@/lib/forecast";

/** How much recent history to draw next to the forecast (a multiple of its length). */
const HISTORY_SHOWN = 2;

interface Row {
  i: number;
  label: string;
  actual?: number;
  forecast?: number;
  band?: [number, number];
}

function ForecastTooltip({
  active,
  payload,
  unit,
}: {
  active?: boolean;
  payload?: { payload: Row }[];
  unit: string;
}) {
  if (!active || !payload?.length) return null;
  const row: Row = payload[0]!.payload;
  return (
    <div className="rounded-md border border-border bg-surface-2 px-3 py-2 font-mono text-[11px] shadow-lg">
      <p className="mb-1 font-semibold text-foreground">{row.label}</p>
      {row.actual !== undefined && (
        <p className="text-foreground">
          <span className="text-muted-foreground">ACTUAL </span>
          {formatValue(row.actual, unit)}
        </p>
      )}
      {row.forecast !== undefined && (
        <>
          <p className="text-foreground">
            <span className="text-muted-foreground">FORECAST </span>
            {formatValue(row.forecast, unit)}
          </p>
          {row.band && (
            <p className="text-muted-foreground">
              80% BAND {formatValue(row.band[0], unit)} –{" "}
              {formatValue(row.band[1], unit)}
            </p>
          )}
        </>
      )}
    </div>
  );
}

function download(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

export function ForecastPanel({ dataset }: { dataset: Dataset }) {
  const [column, setColumn] = useState<string>(dataset.numericNames[0] ?? "");
  const active = dataset.numericNames.includes(column)
    ? column
    : (dataset.numericNames[0] ?? "");

  const outcome = useMemo<{ result?: ForecastResult; error?: string }>(() => {
    if (!active)
      return { error: "This file has no numeric columns to forecast." };
    try {
      return { result: forecastColumn(dataset, active) };
    } catch (e) {
      return {
        error:
          e instanceof Error ? e.message : "Could not forecast this column.",
      };
    }
  }, [dataset, active]);

  const unit = dataset.columns.find((c) => c.name === active)?.unit ?? "";
  const colorIndex = Math.max(0, dataset.numericNames.indexOf(active));
  const { result, error } = outcome;

  const { rows, splitIdx } = useMemo(() => {
    if (!result) return { rows: [] as Row[], splitIdx: 0 };
    const keep = Math.max(24, result.horizon * HISTORY_SHOWN);
    const hist = result.history.slice(-keep);
    const out: Row[] = hist.map((h, i) => ({
      i,
      label: h.label,
      actual: h.value,
    }));
    const lastHist = out.length - 1;
    // Join the lines: the forecast starts at the last real reading.
    const joined = out[lastHist];
    if (joined && joined.actual !== undefined) {
      joined.forecast = joined.actual;
      joined.band = [joined.actual, joined.actual];
    }
    result.points.forEach((p) => {
      out.push({
        i: out.length,
        label: p.label,
        forecast: p.value,
        band: [p.lo, p.hi],
      });
    });
    return { rows: out, splitIdx: lastHist };
  }, [result]);

  const color = chartColor(colorIndex);

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <span className="mr-1 font-mono text-[11px] tracking-wider text-muted-foreground">
          PARAMETER:
        </span>
        {dataset.numericNames.map((name, i) => (
          <button
            key={name}
            onClick={() => setColumn(name)}
            className={`flex items-center gap-1.5 rounded-sm border px-2.5 py-1 font-mono text-[11px] tracking-wide transition-colors ${
              name === active
                ? "border-primary/60 bg-primary/15 text-foreground"
                : "border-border bg-surface text-muted-foreground hover:border-foreground/30 hover:text-foreground"
            }`}
          >
            <span
              className="inline-block size-2 rounded-sm"
              style={{
                background:
                  name === active ? chartColor(i) : "var(--color-border)",
              }}
            />
            {prettyName(name)}
          </button>
        ))}
        {result && (
          <button
            onClick={() =>
              download(`forecast-${active}.csv`, forecastToCsv(result))
            }
            className="ml-auto flex items-center gap-1.5 rounded-sm border border-border bg-surface px-2.5 py-1 font-mono text-[11px] tracking-wider text-foreground transition-colors hover:border-primary/60"
          >
            <Download className="size-3.5" />
            EXPORT FORECAST CSV
          </button>
        )}
      </div>

      {error && (
        <p className="flex items-center gap-2 rounded-md border border-border bg-surface px-4 py-6 font-mono text-xs text-muted-foreground">
          <TriangleAlert className="size-4 shrink-0 text-warning" />
          {error}
        </p>
      )}

      {result && (
        <>
          <div className="mb-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Stat
              label="FORECAST LENGTH"
              value={`${result.horizon} STEPS (= ROWS ÷ 2)`}
            />
            <Stat label="MODEL CHOSEN" value={result.model.label} />
            <Stat
              label="SEASONAL PERIOD"
              value={
                result.seasonalPeriod
                  ? `${result.seasonalPeriod} READINGS`
                  : "NONE FOUND"
              }
            />
            <Stat
              label="VS NAIVE BASELINE"
              value={
                result.baselineRmse === 0
                  ? "—"
                  : `${result.improvementPct >= 0 ? "" : "+"}${(-result.improvementPct).toFixed(0)}% ERROR`
              }
            />
          </div>

          <div className="rounded-md border border-border bg-surface p-2">
            <ResponsiveContainer width="100%" height={340}>
              <ComposedChart
                data={rows}
                margin={{ top: 12, right: 18, bottom: 4, left: 4 }}
              >
                <CartesianGrid
                  stroke="var(--color-border)"
                  strokeOpacity={0.4}
                  strokeDasharray="3 6"
                />
                <XAxis
                  dataKey="i"
                  type="number"
                  domain={[0, Math.max(0, rows.length - 1)]}
                  tickFormatter={(i: number) =>
                    rows[i] ? shortLabel(rows[i].label) : ""
                  }
                  tick={{
                    fill: "var(--color-muted-foreground)",
                    fontSize: 10,
                    fontFamily: "IBM Plex Mono",
                  }}
                  stroke="var(--color-border)"
                  minTickGap={40}
                />
                <YAxis
                  domain={["auto", "auto"]}
                  tick={{
                    fill: "var(--color-muted-foreground)",
                    fontSize: 10,
                    fontFamily: "IBM Plex Mono",
                  }}
                  tickFormatter={(v: number) => formatValue(v)}
                  stroke="var(--color-border)"
                  width={56}
                />
                <Tooltip
                  content={<ForecastTooltip unit={unit} />}
                  cursor={{
                    stroke: "var(--color-primary)",
                    strokeOpacity: 0.4,
                  }}
                />
                <Area
                  dataKey="band"
                  stroke="none"
                  fill={color}
                  fillOpacity={0.18}
                  isAnimationActive={false}
                  connectNulls
                />
                <Line
                  dataKey="actual"
                  stroke={color}
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                  connectNulls={false}
                />
                <Line
                  dataKey="forecast"
                  stroke={color}
                  strokeWidth={2}
                  strokeDasharray="6 4"
                  dot={false}
                  isAnimationActive={false}
                  connectNulls
                />
                <ReferenceLine
                  x={splitIdx}
                  stroke="var(--color-foreground)"
                  strokeOpacity={0.4}
                  strokeDasharray="2 4"
                  label={{
                    value: "NOW",
                    position: "insideTopLeft",
                    fill: "var(--color-muted-foreground)",
                    fontSize: 10,
                    fontFamily: "IBM Plex Mono",
                  }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          <p className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[10px] tracking-wider text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <TrendingUp className="size-3" />
              SOLID = RECORDED · DASHED = FORECAST · SHADED = 80% BAND (NOT A
              GUARANTEE)
            </span>
          </p>

          <div className="mt-4 grid gap-4 lg:grid-cols-[1fr_1fr]">
            <div className="rounded-md border border-border bg-surface p-4">
              <p className="font-mono text-[11px] font-semibold tracking-wider text-foreground">
                MODEL BACK-TEST (LATEST DATA HELD OUT)
              </p>
              <table className="mt-3 w-full font-mono text-[11px]">
                <thead>
                  <tr className="text-left text-muted-foreground">
                    <th className="pb-1 font-medium tracking-wider">MODEL</th>
                    <th className="pb-1 text-right font-medium tracking-wider">
                      RMSE
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {result.candidates.map((c) => (
                    <tr key={c.id} className="border-t border-border/60">
                      <td
                        className={`py-1.5 pr-2 ${c.id === result.model.id ? "font-semibold text-primary" : "text-foreground"}`}
                      >
                        {c.label}
                        {c.id === result.model.id && " ✓"}
                      </td>
                      <td className="py-1.5 text-right text-foreground">
                        {formatValue(c.rmse, unit)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="rounded-md border border-border bg-surface p-4">
              <p className="font-mono text-[11px] font-semibold tracking-wider text-foreground">
                HOW TO READ THIS
              </p>
              <ul className="mt-3 grid gap-1.5 text-xs leading-relaxed text-muted-foreground">
                <li>
                  The model with the lowest error on the most recent data was
                  refitted on all readings and projected {result.horizon} steps
                  ahead.
                </li>
                <li>
                  A forecast continues the patterns in the file. It cannot
                  foresee events, shutdowns or changes in operation.
                </li>
                {result.notes.map((n, i) => (
                  <li key={i}>{n}</li>
                ))}
              </ul>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-surface px-3 py-2">
      <p className="font-mono text-[10px] tracking-wider text-muted-foreground">
        {label}
      </p>
      <p className="mt-0.5 font-mono text-sm font-semibold leading-snug text-foreground">
        {value}
      </p>
    </div>
  );
}
