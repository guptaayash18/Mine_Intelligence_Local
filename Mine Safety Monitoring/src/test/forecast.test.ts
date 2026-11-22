import { describe, expect, it } from "vitest";

import { buildDataset } from "@/lib/dataset";
import {
  detectSeasonalPeriod,
  forecastColumn,
  forecastSeries,
  forecastToCsv,
  interpolateGaps,
} from "@/lib/forecast";

const seasonal = (n: number, m: number, slope = 0) =>
  Array.from(
    { length: n },
    (_, t) => 50 + slope * t + 10 * Math.sin((2 * Math.PI * t) / m),
  );

describe("interpolateGaps", () => {
  it("fills interior and edge gaps", () => {
    const { values, filled } = interpolateGaps([null, 2, null, null, 5, null]);
    expect(values).toEqual([2, 2, 3, 4, 5, 5]);
    expect(filled).toBe(4);
  });
});

describe("detectSeasonalPeriod", () => {
  it("finds a clean period", () => {
    expect(detectSeasonalPeriod(seasonal(240, 24))).toBe(24);
  });
  it("finds none in a straight line", () => {
    expect(
      detectSeasonalPeriod(Array.from({ length: 100 }, (_, t) => 3 + 2 * t)),
    ).toBe(0);
  });
});

describe("forecastSeries", () => {
  it("extends a linear trend", () => {
    const y = Array.from({ length: 60 }, (_, t) => 10 + 2 * t);
    const f = forecastSeries(y, 30);
    expect(f.values).toHaveLength(30);
    expect(f.values[29]!).toBeCloseTo(10 + 2 * 89, 0);
  });

  it("uses a seasonal model on seasonal data and beats the naive baseline", () => {
    const y = seasonal(240, 24, 0.05);
    const f = forecastSeries(y, 120);
    expect(f.seasonalPeriod).toBe(24);
    expect(["seasonal-reg", "holt-winters", "seasonal-naive"]).toContain(
      f.model.id,
    );
    expect(f.improvementPct).toBeGreaterThan(50);
    const truth = seasonal(360, 24, 0.05).slice(240);
    const err = Math.sqrt(
      f.values.reduce((s, v, i) => s + (v - truth[i]!) ** 2, 0) / 120,
    );
    expect(err).toBeLessThan(2);
  });

  it("returns a flat forecast for a constant series", () => {
    const f = forecastSeries(new Array(30).fill(7), 15);
    expect(f.values.every((v) => v === 7)).toBe(true);
    expect(f.lo.every((v) => v === 7)).toBe(true);
  });

  it("keeps the band around the forecast and widening", () => {
    const y = seasonal(200, 20).map((v, i) => v + ((i * 7919) % 13) / 5);
    const f = forecastSeries(y, 100);
    for (let k = 0; k < 100; k++) {
      expect(f.lo[k]!).toBeLessThanOrEqual(f.values[k]!);
      expect(f.hi[k]!).toBeGreaterThanOrEqual(f.values[k]!);
    }
    expect(f.hi[99]! - f.lo[99]!).toBeGreaterThanOrEqual(f.hi[0]! - f.lo[0]!);
  });

  it("refuses very short series", () => {
    expect(() => forecastSeries([1, 2, 3], 2)).toThrow(/at least/);
  });
});

describe("forecastColumn", () => {
  const csv = (n: number) => {
    const lines = ["date,level_m,flow_tph"];
    for (let i = 0; i < n; i++) {
      const d = new Date(Date.UTC(2026, 0, 1) + i * 86_400_000)
        .toISOString()
        .slice(0, 10);
      const flow =
        i === 5
          ? ""
          : (100 + 0.5 * i + 8 * Math.sin((2 * Math.PI * i) / 7)).toFixed(2);
      lines.push(`${d},${(20 + i * 0.1).toFixed(2)},${flow}`);
    }
    return lines.join("\n");
  };

  it("forecasts floor(rows / 2) steps with continuing dates", () => {
    const ds = buildDataset(csv(101), "t.csv");
    const r = forecastColumn(ds, "flow_tph");
    expect(r.horizon).toBe(50);
    expect(r.points).toHaveLength(50);
    expect(r.points[0]!.label).toBe("2026-04-12");
    expect(r.seasonalPeriod).toBe(7);
    expect(r.notes.join(" ")).toMatch(/1 missing value/);
  });

  it("exports CSV with a header and one line per step", () => {
    const ds = buildDataset(csv(40), "t.csv");
    const r = forecastColumn(ds, "level_m");
    const lines = forecastToCsv(r).split("\n");
    expect(lines).toHaveLength(1 + r.horizon);
    expect(lines[0]).toMatch(/^step,time,level_m_forecast/);
  });

  it("explains when a column has too few readings", () => {
    const ds = buildDataset(
      "date,x\n2026-01-01,1\n2026-01-02,2\n2026-01-03,3",
      "t.csv",
    );
    expect(() => forecastColumn(ds, "x")).toThrow(/at least 12/);
  });
});
