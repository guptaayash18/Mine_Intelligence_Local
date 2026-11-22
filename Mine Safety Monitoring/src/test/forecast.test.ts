import { describe, expect, it } from "vitest";

import { axisLabeler, buildDataset, shortLabel } from "@/lib/dataset";
import {
  detectSeasonalPeriod,
  forecastColumn,
  forecastSeries,
  forecastToCsv,
  interpolateGaps,
  simulatePath,
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

describe("axisLabeler", () => {
  const hourly = (days: number) =>
    Array.from({ length: days * 24 }, (_, i) => {
      const d = new Date(Date.UTC(2026, 9, 22, 0) + i * 3_600_000);
      return d.toISOString().slice(0, 19).replace("T", " ");
    });

  it("shows time of day for hourly data on one day", () => {
    const f = axisLabeler(hourly(1).slice(0, 12));
    expect(f("2026-10-22 05:00:00")).toMatch(/05[:.]00/);
    expect(f("2026-10-22 05:00:00")).not.toMatch(/Oct/);
  });
  it("shows date and time when sub-daily data crosses midnight", () => {
    const f = axisLabeler(hourly(2));
    expect(f("2026-10-23 05:00:00")).toMatch(/23.*05[:.]00|Oct.*23.*05[:.]00/);
  });
  it("falls back to the date for long or date-only data", () => {
    const daily = Array.from(
      { length: 30 },
      (_, i) => `2026-10-${String(i + 1).padStart(2, "0")}`,
    );
    expect(axisLabeler(daily)("2026-10-05")).toBe(shortLabel("2026-10-05"));
    expect(axisLabeler(hourly(10))("2026-10-05 05:00:00")).toBe(
      shortLabel("2026-10-05 05:00:00"),
    );
  });
});

describe("simulated scenario", () => {
  // Trend plus smooth-wandering noise, like a real sensor log.
  const noisy = (n: number) => {
    let e = 0;
    let s = 12345;
    const rnd = () =>
      ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296 - 0.5) * 2;
    return Array.from({ length: n }, (_, t) => {
      e = 0.6 * e + rnd() * 0.8;
      return 100 + 0.4 * t + e;
    });
  };
  const stepStd = (a: number[]) => {
    const d = a.slice(1).map((v, i) => v - a[i]!);
    const m = d.reduce((x, y) => x + y, 0) / d.length;
    return Math.sqrt(d.reduce((x, y) => x + (y - m) ** 2, 0) / d.length);
  };

  it("varies from step to step instead of following a straight line", () => {
    const y = noisy(300);
    const f = forecastSeries(y, 150);
    const wiggle = stepStd(f.sim);
    const mean = stepStd(f.values);
    expect(mean).toBeLessThan(0.05); // expected path is (almost) straight
    expect(wiggle).toBeGreaterThan(0.1);
  });

  it("has step-to-step roughness comparable to the recorded data", () => {
    const y = noisy(300);
    const f = forecastSeries(y, 150);
    // Remove the trend from history and from the scenario, then compare roughness.
    const detrend = (a: number[]) => a.map((v, i) => v - 0.4 * i);
    const hist = stepStd(detrend(y));
    const sim = stepStd(f.sim.map((v, i) => v - f.values[i]!));
    expect(sim / hist).toBeGreaterThan(0.5);
    expect(sim / hist).toBeLessThan(2);
  });

  it("mostly stays inside the 80% band", () => {
    const f = forecastSeries(noisy(300), 150);
    const inside = f.sim.filter(
      (v, i) => v >= f.lo[i]! && v <= f.hi[i]!,
    ).length;
    expect(inside / f.sim.length).toBeGreaterThan(0.6);
  });

  it("is repeatable per seed and differs between seeds", () => {
    const f = forecastSeries(noisy(200), 100);
    const a = simulatePath(f.values, f.sigma, f.noisePhi, f.lastResidual, 7);
    const b = simulatePath(f.values, f.sigma, f.noisePhi, f.lastResidual, 7);
    const c = simulatePath(f.values, f.sigma, f.noisePhi, f.lastResidual, 8);
    expect(a).toEqual(b);
    expect(a).not.toEqual(c);
  });

  it("stays flat for a constant series", () => {
    const f = forecastSeries(new Array(30).fill(5), 10);
    expect(f.sim.every((v) => v === 5)).toBe(true);
  });
});
