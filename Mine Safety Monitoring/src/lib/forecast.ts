/**
 * Time-series forecasting, computed in the browser (no network, no extra
 * packages). Given a numeric column, it predicts the next half of the series:
 * a file with n rows gets a forecast of floor(n / 2) steps.
 *
 * Many models compete on a hold-out of the most recent data and the one with
 * the lowest error is refitted on everything and used:
 *
 *   naive            repeat the last value (the baseline to beat)
 *   linear           straight-line regression on time
 *   holt             Holt exponential smoothing with a damped trend
 *   seasonal-naive   repeat the last full season
 *   seasonal-reg     linear trend + a fixed average pattern per season position
 *   holt-winters     damped-trend Holt-Winters with an additive seasonal part
 *   ses, theta, quadratic, arima, sarima, neural   see forecast-models.ts
 *   ensemble         the average of the three best models above
 *
 * The seasonal period is found from the autocorrelation of the detrended
 * series. Prediction bands are empirical 80% bands built from the model's
 * own one-step errors and its hold-out error. They are not guarantees.
 */
import {
  chartOrderedRows,
  type Dataset,
  type ForecastSummary,
} from "@/lib/dataset";
import { EXTRA_MODELS, type ExtraModelId } from "@/lib/forecast-models";
import { medianSamplingIntervalMs, parseTimestamp } from "@/lib/moving-average";

/* ============================ Types ============================ */

export type ForecastModelId =
  | "naive"
  | "linear"
  | "holt"
  | "seasonal-naive"
  | "seasonal-reg"
  | "holt-winters"
  | ExtraModelId
  | "ensemble";

export interface ForecastCandidate {
  id: ForecastModelId;
  label: string;
  /** Hold-out root-mean-square error, in the column's own unit. */
  rmse: number;
}

export interface ForecastPoint {
  /** 1-based step after the last observation. */
  step: number;
  label: string;
  /** Epoch ms, or null when the file has no usable time column. */
  ts: number | null;
  /** Expected value: the model's best single guess (smooth by nature). */
  value: number;
  lo: number;
  hi: number;
  /** Spread (one standard deviation) of the forecast error at this step. */
  sigma: number;
  /** One simulated scenario: the expected value plus realistic variation. */
  sim: number;
}

export interface HistoryPoint {
  label: string;
  value: number;
}

export interface ForecastResult {
  column: string;
  /** Number of rows the forecast length was derived from. */
  rowCount: number;
  horizon: number;
  /** Median sampling interval, or null without a time column. */
  stepMs: number | null;
  /** Seasonal period in rows, 0 when none was detected. */
  seasonalPeriod: number;
  model: { id: ForecastModelId; label: string };
  candidates: ForecastCandidate[];
  /** Hold-out RMSE of the naive baseline, and how much the chosen model beats it. */
  baselineRmse: number;
  improvementPct: number;
  /** Width of the empirical band (always 80). */
  bandPct: 80;
  /** Step-to-step persistence of the model's errors (0 = jagged, near 1 = smooth wander). */
  noisePhi: number;
  /** Starting offset for the simulation: how far the last reading sits from the fit. */
  lastResidual: number;
  history: HistoryPoint[];
  points: ForecastPoint[];
  notes: string[];
}

/* ============================ Settings ============================ */

export const MIN_POINTS = 12;
/** Only the most recent values are used to fit; older history adds cost, not accuracy. */
const MAX_FIT_POINTS = 4000;
const MAX_HORIZON = 5000;
const MAX_SEASON = 400;
const Z80 = 1.2816;

/* ============================ Small helpers ============================ */

const mean = (a: readonly number[]) =>
  a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0;

function std(a: readonly number[]): number {
  if (a.length < 2) return 0;
  const m = mean(a);
  return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1));
}

function rmse(actual: readonly number[], pred: readonly number[]): number {
  const n = Math.min(actual.length, pred.length);
  if (!n) return Number.POSITIVE_INFINITY;
  let s = 0;
  for (let i = 0; i < n; i++) s += (actual[i]! - pred[i]!) ** 2;
  return Math.sqrt(s / n);
}

/** Ordinary least squares of y on t = 0..n-1. */
function ols(y: readonly number[]): { a: number; b: number } {
  const n = y.length;
  if (n === 0) return { a: 0, b: 0 };
  const tm = (n - 1) / 2;
  const ym = mean(y);
  let num = 0;
  let den = 0;
  for (let t = 0; t < n; t++) {
    num += (t - tm) * (y[t]! - ym);
    den += (t - tm) ** 2;
  }
  const b = den === 0 ? 0 : num / den;
  return { a: ym - b * tm, b };
}

/**
 * Fill gaps by straight-line interpolation (edges copy the nearest value).
 * Returns the filled series and how many values were invented.
 */
export function interpolateGaps(raw: readonly (number | null)[]): {
  values: number[];
  filled: number;
} {
  const out: number[] = new Array(raw.length).fill(0);
  let filled = 0;
  let prev = -1;
  for (let i = 0; i < raw.length; i++) {
    const v = raw[i];
    if (typeof v === "number" && Number.isFinite(v)) {
      if (prev === -1) {
        for (let j = 0; j < i; j++) out[j] = v;
        filled += i;
      } else if (i - prev > 1) {
        const p = out[prev]!;
        for (let j = prev + 1; j < i; j++) {
          out[j] = p + ((v - p) * (j - prev)) / (i - prev);
        }
        filled += i - prev - 1;
      }
      out[i] = v;
      prev = i;
    }
  }
  if (prev !== -1) {
    for (let j = prev + 1; j < raw.length; j++) out[j] = out[prev]!;
    filled += raw.length - 1 - prev;
  }
  return { values: out, filled };
}

/* ============================ Season detection ============================ */

/**
 * Seasonal period in samples, or 0. Uses the autocorrelation of the series
 * after removing a straight-line trend; the period must repeat at least
 * three times and show a clear peak (autocorrelation >= 0.3).
 */
export function detectSeasonalPeriod(y: readonly number[]): number {
  const n = y.length;
  const maxLag = Math.min(Math.floor(n / 3), MAX_SEASON);
  if (maxLag < 2) return 0;
  const { a, b } = ols(y);
  const r = y.map((v, t) => v - (a + b * t));
  const rm = mean(r);
  const c = r.map((v) => v - rm);
  let c0 = 0;
  for (const v of c) c0 += v * v;
  if (c0 === 0) return 0;

  const acf: number[] = new Array(maxLag + 2).fill(0);
  for (let lag = 1; lag <= maxLag + 1; lag++) {
    let s = 0;
    for (let t = lag; t < n; t++) s += c[t]! * c[t - lag]!;
    acf[lag] = s / c0;
  }

  let best = 0;
  let bestVal = 0.3;
  for (let lag = 2; lag <= maxLag; lag++) {
    const isPeak = acf[lag]! > acf[lag - 1]! && acf[lag]! >= acf[lag + 1]!;
    if (isPeak && acf[lag]! > bestVal) {
      best = lag;
      bestVal = acf[lag]!;
    }
  }
  if (!best) return 0;
  // Prefer the shortest lag that is nearly as strong (avoids picking 2x, 3x the period).
  for (let lag = 2; lag < best; lag++) {
    const isPeak = acf[lag]! > acf[lag - 1]! && acf[lag]! >= acf[lag + 1]!;
    if (isPeak && acf[lag]! >= 0.9 * bestVal) return lag;
  }
  return best;
}

/* ============================ Models ============================ */

interface Fitted {
  forecast: (h: number) => number[];
  /** One-step in-sample errors (actual − fitted), early warm-up values dropped. */
  residuals: number[];
}

interface Model {
  id: ForecastModelId;
  label: string;
  needsSeason: boolean;
  fit: (y: readonly number[], m: number) => Fitted | null;
}

const naive: Model = {
  id: "naive",
  label: "Naive (last value)",
  needsSeason: false,
  fit: (y) => {
    const last = y[y.length - 1]!;
    const residuals: number[] = [];
    for (let t = 1; t < y.length; t++) residuals.push(y[t]! - y[t - 1]!);
    return { forecast: (h) => new Array(h).fill(last), residuals };
  },
};

const linear: Model = {
  id: "linear",
  label: "Linear regression",
  needsSeason: false,
  fit: (y) => {
    const { a, b } = ols(y);
    const n = y.length;
    const residuals = y.map((v, t) => v - (a + b * t));
    return {
      forecast: (h) => Array.from({ length: h }, (_, k) => a + b * (n + k)),
      residuals,
    };
  },
};

const DAMP = [0.8, 0.9, 0.97, 1];

const holt: Model = {
  id: "holt",
  label: "Holt exponential smoothing (damped trend)",
  needsSeason: false,
  fit: (y) => {
    const n = y.length;
    if (n < 4) return null;
    let best: {
      sse: number;
      l: number;
      b: number;
      phi: number;
      res: number[];
    } | null = null;
    for (const alpha of [0.1, 0.3, 0.5, 0.8]) {
      for (const beta of [0.01, 0.05, 0.15, 0.3]) {
        for (const phi of DAMP) {
          let l = y[0]!;
          let b = (y[Math.min(3, n - 1)]! - y[0]!) / Math.min(3, n - 1);
          let sse = 0;
          const res: number[] = [];
          for (let t = 1; t < n; t++) {
            const pred = l + phi * b;
            const e = y[t]! - pred;
            sse += e * e;
            res.push(e);
            const lNew = alpha * y[t]! + (1 - alpha) * pred;
            b = beta * (lNew - l) + (1 - beta) * phi * b;
            l = lNew;
          }
          if (!best || sse < best.sse) best = { sse, l, b, phi, res };
        }
      }
    }
    if (!best) return null;
    const { l, b, phi, res } = best;
    return {
      residuals: res,
      forecast: (h) => {
        const out: number[] = [];
        let damp = 0;
        let p = 1;
        for (let k = 1; k <= h; k++) {
          p *= phi;
          damp += p;
          out.push(l + damp * b);
        }
        return out;
      },
    };
  },
};

const seasonalNaive: Model = {
  id: "seasonal-naive",
  label: "Seasonal naive (repeat last season)",
  needsSeason: true,
  fit: (y, m) => {
    const n = y.length;
    if (m < 2 || n < 2 * m) return null;
    const residuals: number[] = [];
    for (let t = m; t < n; t++) residuals.push(y[t]! - y[t - m]!);
    return {
      residuals,
      forecast: (h) => Array.from({ length: h }, (_, k) => y[n - m + (k % m)]!),
    };
  },
};

const seasonalReg: Model = {
  id: "seasonal-reg",
  label: "Seasonal regression (trend + seasonal pattern)",
  needsSeason: true,
  fit: (y, m) => {
    const n = y.length;
    if (m < 2 || n < 2 * m) return null;
    const s: number[] = new Array(m).fill(0);
    let a = 0;
    let b = 0;
    // Backfitting: alternate between the trend and the seasonal pattern.
    for (let iter = 0; iter < 5; iter++) {
      ({ a, b } = ols(y.map((v, t) => v - s[t % m]!)));
      const sums: number[] = new Array(m).fill(0);
      const counts: number[] = new Array(m).fill(0);
      for (let t = 0; t < n; t++) {
        sums[t % m]! += y[t]! - (a + b * t);
        counts[t % m]! += 1;
      }
      const raw = sums.map((v, j) => v / (counts[j] || 1));
      const centre = mean(raw);
      for (let j = 0; j < m; j++) s[j] = raw[j]! - centre;
    }
    const residuals = y.map((v, t) => v - (a + b * t + s[t % m]!));
    return {
      residuals,
      forecast: (h) =>
        Array.from({ length: h }, (_, k) => a + b * (n + k) + s[(n + k) % m]!),
    };
  },
};

const holtWinters: Model = {
  id: "holt-winters",
  label: "Holt-Winters (additive seasonality, damped trend)",
  needsSeason: true,
  fit: (y, m) => {
    const n = y.length;
    if (m < 2 || n < 2 * m) return null;
    const first = mean(y.slice(0, m));
    const second = mean(y.slice(m, 2 * m));
    const l0 = first;
    const b0 = (second - first) / m;
    const s0 = Array.from({ length: m }, (_, i) => y[i]! - first);

    let best: {
      sse: number;
      l: number;
      b: number;
      s: number[];
      phi: number;
      res: number[];
    } | null = null;

    for (const alpha of [0.1, 0.3, 0.5, 0.8]) {
      for (const beta of [0.01, 0.05, 0.15]) {
        for (const gamma of [0.05, 0.2, 0.5]) {
          for (const phi of [0.9, 0.98, 1]) {
            let l = l0;
            let b = b0;
            const s = s0.slice();
            let sse = 0;
            const res: number[] = [];
            for (let t = 0; t < n; t++) {
              const si = t % m;
              const base = l + phi * b;
              const pred = base + s[si]!;
              const e = y[t]! - pred;
              if (t >= m) {
                sse += e * e;
                res.push(e);
              }
              const lNew = alpha * (y[t]! - s[si]!) + (1 - alpha) * base;
              b = beta * (lNew - l) + (1 - beta) * phi * b;
              s[si] = gamma * (y[t]! - base) + (1 - gamma) * s[si]!;
              l = lNew;
            }
            if (!best || sse < best.sse) best = { sse, l, b, s, phi, res };
          }
        }
      }
    }
    if (!best) return null;
    const { l, b, s, phi, res } = best;
    return {
      residuals: res,
      forecast: (h) => {
        const out: number[] = [];
        let damp = 0;
        let p = 1;
        for (let k = 1; k <= h; k++) {
          p *= phi;
          damp += p;
          out.push(l + damp * b + s[(n + k - 1) % m]!);
        }
        return out;
      },
    };
  },
};

const MODELS: Model[] = [
  ...EXTRA_MODELS,
  naive,
  linear,
  holt,
  seasonalNaive,
  seasonalReg,
  holtWinters,
];

/** Average of several models: forecasts and in-sample errors are averaged. */
function makeEnsemble(parts: Model[]): Model {
  return {
    id: "ensemble",
    label: `Ensemble (average of ${parts.length} best models)`,
    needsSeason: false,
    fit: (y, m) => {
      const fits = parts
        .map((p) => p.fit(y, m))
        .filter((f): f is Fitted => f !== null);
      if (fits.length === 0) return null;
      const len = Math.min(...fits.map((f) => f.residuals.length));
      return {
        // Errors are aligned from the most recent end.
        residuals: Array.from({ length: len }, (_, i) =>
          mean(fits.map((f) => f.residuals[f.residuals.length - len + i]!)),
        ),
        forecast: (h) => {
          const all = fits.map((f) => f.forecast(h));
          return Array.from({ length: h }, (_, k) =>
            mean(all.map((a) => a[k]!)),
          );
        },
      };
    },
  };
}

/* ============================ Core forecast ============================ */

export interface CoreForecast {
  model: { id: ForecastModelId; label: string };
  candidates: ForecastCandidate[];
  seasonalPeriod: number;
  values: number[];
  lo: number[];
  hi: number[];
  sigma: number[];
  noisePhi: number;
  lastResidual: number;
  sim: number[];
  baselineRmse: number;
  improvementPct: number;
}

/* ============================ Simulated scenario ============================ */

export const DEFAULT_SEED = 1;

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Lag-1 autocorrelation of the errors, clipped to a stable range. */
function lag1Autocorrelation(r: readonly number[]): number {
  if (r.length < 3) return 0;
  const m = mean(r);
  let num = 0;
  let den = 0;
  for (let t = 0; t < r.length; t++) {
    den += (r[t]! - m) ** 2;
    if (t > 0) num += (r[t]! - m) * (r[t - 1]! - m);
  }
  if (den === 0) return 0;
  return Math.max(-0.5, Math.min(0.95, num / den));
}

/**
 * One plausible future: the expected-value path plus noise that behaves like
 * the model's own past errors. The noise is a first-order autoregression, so
 * it wanders as smoothly (or as jaggedly) as the file's own data does, and
 * its spread at every step equals the forecast spread, so about 80% of the
 * path stays inside the shaded band. The same seed always gives the same path.
 */
export function simulatePath(
  values: readonly number[],
  sigma: readonly number[],
  phi: number,
  lastResidual: number,
  seed: number,
): number[] {
  const rand = mulberry32(seed * 2654435761 + 12345);
  const gauss = () => {
    const u = Math.max(rand(), 1e-12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
  };
  const innov = Math.sqrt(Math.max(0, 1 - phi * phi));
  const s0 = sigma[0] ?? 0;
  // Start from where the data last sat relative to its fit, so the scenario
  // continues from the recorded line instead of jumping.
  let prev = s0 > 0 ? Math.max(-2 * s0, Math.min(2 * s0, lastResidual)) : 0;
  const out: number[] = [];
  for (let k = 0; k < values.length; k++) {
    const sk = sigma[k] ?? 0;
    const sPrev = k === 0 ? s0 : (sigma[k - 1] ?? 0);
    // Keep the standardised noise level continuous when sigma grows.
    const z = sPrev > 0 ? prev / sPrev : 0;
    const zNew = phi * z + innov * gauss();
    prev = zNew * sk;
    out.push(values[k]! + prev);
  }
  return out;
}

/**
 * Forecast `h` steps of an evenly spaced, gap-free series. Picks the best
 * model on a hold-out of the latest data, then refits it on everything.
 */
export function forecastSeries(y: readonly number[], h: number): CoreForecast {
  const n = y.length;
  if (n < MIN_POINTS) {
    throw new Error(
      `Need at least ${MIN_POINTS} valid readings to forecast; this column has ${n}.`,
    );
  }
  if (!(h >= 1)) throw new Error("Forecast length must be at least 1.");

  // A flat series has nothing to learn.
  if (std(y) === 0) {
    const v = y[0]!;
    return {
      model: { id: "naive", label: naive.label },
      candidates: [{ id: "naive", label: naive.label, rmse: 0 }],
      seasonalPeriod: 0,
      values: new Array(h).fill(v),
      lo: new Array(h).fill(v),
      hi: new Array(h).fill(v),
      sigma: new Array(h).fill(0),
      noisePhi: 0,
      lastResidual: 0,
      sim: new Array(h).fill(v),
      baselineRmse: 0,
      improvementPct: 0,
    };
  }

  const m = detectSeasonalPeriod(y);
  const nVal = Math.max(2, Math.min(h, Math.floor(n / 3)));
  const train = y.slice(0, n - nVal);
  const valid = y.slice(n - nVal);

  const scored: { model: Model; rmse: number; pred: number[] }[] = [];
  for (const model of MODELS) {
    if (model.needsSeason && m < 2) continue;
    const fit = model.fit(train, m);
    if (!fit) continue;
    const pred = fit.forecast(nVal);
    const e = rmse(valid, pred);
    if (Number.isFinite(e)) scored.push({ model, rmse: e, pred });
  }
  scored.sort((p, q) => p.rmse - q.rmse);

  // Ensemble: the average of the three best models is often better than any
  // one of them. It competes like the rest.
  const top = scored.slice(0, 3);
  if (top.length >= 2) {
    const pred = Array.from({ length: nVal }, (_, i) =>
      mean(top.map((t) => t.pred[i]!)),
    );
    const e = rmse(valid, pred);
    if (Number.isFinite(e)) {
      scored.push({
        model: makeEnsemble(top.map((t) => t.model)),
        rmse: e,
        pred,
      });
      scored.sort((p, q) => p.rmse - q.rmse);
    }
  }
  const winner = scored[0]!.model;
  const baseline = scored.find((s) => s.model.id === "naive")!.rmse;
  const winRmse = scored[0]!.rmse;

  const full = winner.fit(y, m) ?? naive.fit(y, m)!;
  const values = full.forecast(h);

  // Error grows from the one-step error to the hold-out error over the
  // hold-out length, then stays at the hold-out level.
  const sigmaIn = std(full.residuals);
  const sigmaVal = Math.max(sigmaIn, winRmse);
  const lo: number[] = [];
  const hi: number[] = [];
  const sigmas: number[] = [];
  for (let k = 1; k <= h; k++) {
    const frac = Math.min(1, (k - 1) / Math.max(1, nVal - 1));
    const sigma = Math.sqrt(
      sigmaIn ** 2 + frac * (sigmaVal ** 2 - sigmaIn ** 2),
    );
    sigmas.push(sigma);
    lo.push(values[k - 1]! - Z80 * sigma);
    hi.push(values[k - 1]! + Z80 * sigma);
  }

  const noisePhi = lag1Autocorrelation(full.residuals);
  const lastResidual = full.residuals[full.residuals.length - 1] ?? 0;

  return {
    model: { id: winner.id, label: winner.label },
    candidates: scored.map((s) => ({
      id: s.model.id,
      label: s.model.label,
      rmse: s.rmse,
    })),
    seasonalPeriod: m,
    values,
    lo,
    hi,
    sigma: sigmas,
    noisePhi,
    lastResidual,
    sim: simulatePath(values, sigmas, noisePhi, lastResidual, DEFAULT_SEED),
    baselineRmse: baseline,
    improvementPct: baseline > 0 ? ((baseline - winRmse) / baseline) * 100 : 0,
  };
}

/* ============================ Dataset wrapper ============================ */

const pad = (v: number, w = 2) => String(v).padStart(w, "0");

/** Format a future timestamp in the same style as the file's own labels. */
export function formatFutureLabel(ts: number, sample: string): string {
  const d = new Date(ts);
  if (/^\d{4}-\d{2}-\d{2}$/.test(sample.trim())) {
    return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}`;
  }
  const date = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  const time = `${pad(d.getHours())}:${pad(d.getMinutes())}`;
  const withSeconds = /\d{1,2}:\d{2}:\d{2}/.test(sample);
  return `${date} ${time}${withSeconds ? `:${pad(d.getSeconds())}` : ""}`;
}

/** Forecast one numeric column of a dataset for floor(rowCount / 2) steps. */
export function forecastColumn(
  dataset: Dataset,
  column: string,
): ForecastResult {
  const rows = chartOrderedRows(dataset);
  const tc = dataset.timeColumn;
  const raw = rows.map((r) =>
    typeof r[column] === "number" ? (r[column] as number) : null,
  );
  const labels = rows.map((r, i) => (tc ? String(r[tc] ?? `#${i}`) : `#${i}`));
  const times = tc
    ? rows.map((r) => parseTimestamp(r[tc]))
    : rows.map(() => null);

  const validCount = raw.filter((v) => v !== null).length;
  if (validCount < MIN_POINTS) {
    throw new Error(
      `Need at least ${MIN_POINTS} valid readings to forecast; this column has ${validCount}.`,
    );
  }

  const notes: string[] = [];
  const { values: filledAll, filled } = interpolateGaps(raw);
  if (filled > 0) {
    notes.push(
      `${filled} missing value${filled === 1 ? "" : "s"} filled by straight-line interpolation before fitting.`,
    );
  }

  const rowCount = rows.length;
  let horizon = Math.floor(rowCount / 2);
  if (horizon > MAX_HORIZON) {
    horizon = MAX_HORIZON;
    notes.push(`Forecast length capped at ${MAX_HORIZON} steps.`);
  }
  if (horizon < 1) horizon = 1;

  let y = filledAll;
  let fitLabels = labels;
  let fitTimes = times;
  if (y.length > MAX_FIT_POINTS) {
    const cut = y.length - MAX_FIT_POINTS;
    y = y.slice(cut);
    fitLabels = labels.slice(cut);
    fitTimes = times.slice(cut);
    notes.push(`Fitted on the most recent ${MAX_FIT_POINTS} readings.`);
  }

  // Step size and regularity of the time axis.
  const stepMs = tc ? medianSamplingIntervalMs(fitTimes) : null;
  if (tc && stepMs && stepMs > 0) {
    let irregular = 0;
    for (let i = 1; i < fitTimes.length; i++) {
      const a = fitTimes[i - 1];
      const b = fitTimes[i];
      if (a === null || b === null || Math.abs(b! - a! - stepMs) > stepMs * 0.5)
        irregular++;
    }
    if (irregular / fitTimes.length > 0.1) {
      notes.push(
        "Timestamps are not evenly spaced; readings are treated as equally spaced in row order.",
      );
    }
  } else if (tc) {
    notes.push(
      "Could not work out the time step; forecast steps are labelled by row count.",
    );
  }

  const core = forecastSeries(y, horizon);

  const lastTs =
    [...fitTimes].reverse().find((t): t is number => t !== null) ?? null;
  const sample = fitLabels[fitLabels.length - 1] ?? "";
  const canDate = !!(tc && stepMs && stepMs > 0 && lastTs !== null);
  const points: ForecastPoint[] = core.values.map((value, i) => {
    const step = i + 1;
    const ts = canDate ? lastTs! + step * stepMs! : null;
    return {
      step,
      ts,
      label: ts !== null ? formatFutureLabel(ts, sample) : `+${step}`,
      value,
      lo: core.lo[i]!,
      hi: core.hi[i]!,
      sigma: core.sigma[i]!,
      sim: core.sim[i]!,
    };
  });

  if (core.seasonalPeriod > 0 && canDate) {
    const spanH = (core.seasonalPeriod * stepMs!) / 3_600_000;
    notes.push(
      spanH >= 48
        ? `Repeating pattern found every ${core.seasonalPeriod} readings (about ${(spanH / 24).toFixed(1)} days).`
        : `Repeating pattern found every ${core.seasonalPeriod} readings (about ${spanH.toFixed(spanH < 10 ? 1 : 0)} hours).`,
    );
  } else if (core.seasonalPeriod > 0) {
    notes.push(
      `Repeating pattern found every ${core.seasonalPeriod} readings.`,
    );
  }

  return {
    column,
    rowCount,
    horizon,
    stepMs: canDate ? stepMs : null,
    seasonalPeriod: core.seasonalPeriod,
    model: core.model,
    candidates: core.candidates,
    baselineRmse: core.baselineRmse,
    improvementPct: core.improvementPct,
    bandPct: 80,
    noisePhi: core.noisePhi,
    lastResidual: core.lastResidual,
    history: y.map((value, i) => ({ label: fitLabels[i] ?? `#${i}`, value })),
    points,
    notes,
  };
}

/** CSV text of a forecast: step, time, forecast, lower and upper bound. */
export function forecastToCsv(result: ForecastResult): string {
  const q = (s: string) =>
    /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  const lines = [
    `step,time,${q(result.column)}_forecast,${q(result.column)}_lower_80,${q(result.column)}_upper_80,${q(result.column)}_simulated`,
  ];
  for (const p of result.points) {
    lines.push(
      [
        p.step,
        q(p.label),
        p.value.toFixed(4),
        p.lo.toFixed(4),
        p.hi.toFixed(4),
        p.sim.toFixed(4),
      ].join(","),
    );
  }
  return lines.join("\n");
}

/* ============================ Summary for the AI report ============================ */

const SUMMARY_COLUMNS = 20;

/** A 0/1 detection flag: a forecast of it would say nothing useful. */
function isFlagColumn(dataset: Dataset, name: string): boolean {
  let seen = 0;
  for (const row of dataset.rows) {
    const v = row[name];
    if (typeof v !== "number") continue;
    if (v !== 0 && v !== 1) return false;
    seen++;
  }
  return seen > 0;
}

/**
 * Forecast every numeric column (up to 20) and keep only what the AI report
 * needs: where each one is expected to be at the end and the 80% range.
 * Columns that cannot be forecast (too few readings) are left out.
 */
export function buildForecastSummary(
  dataset: Dataset,
): ForecastSummary | undefined {
  const items: ForecastSummary["items"] = [];
  let horizon = 0;
  let until: string | null = null;
  for (const name of dataset.numericNames.slice(0, SUMMARY_COLUMNS)) {
    if (isFlagColumn(dataset, name)) continue;
    let r: ForecastResult;
    try {
      r = forecastColumn(dataset, name);
    } catch {
      continue;
    }
    const lastPoint = r.points[r.points.length - 1];
    const last = r.history[r.history.length - 1]?.value;
    if (!lastPoint || last === undefined) continue;
    horizon = r.horizon;
    until = r.stepMs !== null ? lastPoint.label : until;
    const unit = dataset.columns.find((c) => c.name === name)?.unit ?? "";
    items.push({
      column: name,
      unit,
      model: r.model.label,
      last: Number(last.toPrecision(6)),
      end: Number(lastPoint.value.toPrecision(6)),
      changePct:
        last !== 0
          ? Number(
              (((lastPoint.value - last) / Math.abs(last)) * 100).toFixed(1),
            )
          : 0,
      lo: Number(lastPoint.lo.toPrecision(6)),
      hi: Number(lastPoint.hi.toPrecision(6)),
    });
  }
  return items.length ? { horizon, until, items } : undefined;
}

/** Below this change, the expected value is described as roughly steady. */
export const OUTLOOK_FLAT_PCT = 2;

const withUnit = (v: number, unit: string) =>
  `${Number(v.toPrecision(4))}${unit ? ` ${unit}` : ""}`;

/** One plain sentence per parameter, written by code (never by the model). */
export function outlookText(item: ForecastSummary["items"][number]): string {
  const move =
    Math.abs(item.changePct) < OUTLOOK_FLAT_PCT
      ? "expected to stay roughly level"
      : item.changePct > 0
        ? `expected to rise about ${Math.abs(item.changePct)}%`
        : `expected to fall about ${Math.abs(item.changePct)}%`;
  return (
    `${move}, from ${withUnit(item.last, item.unit)} to about ` +
    `${withUnit(item.end, item.unit)} (80% range ${withUnit(item.lo, "")} to ` +
    `${withUnit(item.hi, item.unit)}). Model: ${item.model}.`
  );
}

/** Outlook lines for the parameters the report already covers. */
export function planOutlook(
  summary: ForecastSummary | undefined,
  parameters: readonly string[],
): { parameter: string; text: string }[] {
  if (!summary) return [];
  const out: { parameter: string; text: string }[] = [];
  for (const p of parameters) {
    const item = summary.items.find((i) => i.column === p);
    if (item) out.push({ parameter: p, text: outlookText(item) });
  }
  return out;
}
