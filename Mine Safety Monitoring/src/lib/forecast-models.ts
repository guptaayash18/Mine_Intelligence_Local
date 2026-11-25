/**
 * Extra forecasting models that compete with the core ones in forecast.ts.
 * All run in the browser with no packages and are deterministic (the neural
 * network uses a fixed seed). Each fits a gap-free, evenly spaced series and
 * returns a forecast function plus its one-step in-sample errors.
 *
 *   ses        simple exponential smoothing (flat level)
 *   theta      Theta method: smoothing plus half the straight-line drift
 *   quadratic  degree-2 polynomial trend (clamped so it cannot run away)
 *   arima      ARIMA(p, d, 0): autoregression on the series or its first
 *              difference; p and d chosen by AICc
 *   sarima     SARIMA(p, 0, 0)(0, 1, 0)m: seasonal difference + autoregression
 *   neural     small neural network (one hidden layer) on recent values,
 *              the seasonal phase and a removed trend
 */

export type ExtraModelId =
  "ses" | "theta" | "quadratic" | "arima" | "sarima" | "neural";

export interface ExtraFitted {
  forecast: (h: number) => number[];
  /** One-step in-sample errors (actual − fitted), warm-up values dropped. */
  residuals: number[];
}

export interface ExtraModel {
  id: ExtraModelId;
  label: string;
  needsSeason: boolean;
  fit: (y: readonly number[], m: number) => ExtraFitted | null;
}

/* ============================ Maths helpers ============================ */

const mean = (a: readonly number[]) =>
  a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0;

function olsLine(y: readonly number[]): { a: number; b: number } {
  const n = y.length;
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

/** Solve A x = b by Gaussian elimination with partial pivoting (small systems). */
function solve(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]!]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++)
      if (Math.abs(M[r]![c]!) > Math.abs(M[p]![c]!)) p = r;
    if (Math.abs(M[p]![c]!) < 1e-12) return null;
    [M[c], M[p]] = [M[p]!, M[c]!];
    for (let r = c + 1; r < n; r++) {
      const f = M[r]![c]! / M[c]![c]!;
      for (let k = c; k <= n; k++) M[r]![k]! -= f * M[c]![k]!;
    }
  }
  const x: number[] = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r]![n]!;
    for (let k = r + 1; k < n; k++) s -= M[r]![k]! * x[k]!;
    x[r] = s / M[r]![r]!;
  }
  return x;
}

/**
 * Least-squares autoregression z_t = c + φ1 z_{t-1} + … + φp z_{t-p}, fitted
 * on rows t = start … n-1. Returns coefficients and one-step residuals.
 */
function fitAr(
  z: readonly number[],
  p: number,
  start: number,
): { coef: number[]; residuals: number[]; rss: number } | null {
  const rows = z.length - start;
  if (rows < p + 3 || start < p) return null;
  const k = p + 1;
  const XtX: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
  const Xty: number[] = new Array(k).fill(0);
  const x: number[] = new Array(k).fill(0);
  for (let t = start; t < z.length; t++) {
    x[0] = 1;
    for (let j = 1; j <= p; j++) x[j] = z[t - j]!;
    for (let a = 0; a < k; a++) {
      Xty[a]! += x[a]! * z[t]!;
      for (let b = 0; b < k; b++) XtX[a]![b]! += x[a]! * x[b]!;
    }
  }
  // A whisper of ridge keeps near-collinear lags solvable.
  for (let a = 1; a < k; a++) XtX[a]![a]! += 1e-8 * (XtX[a]![a]! + 1);
  const coef = solve(XtX, Xty);
  if (!coef) return null;
  const residuals: number[] = [];
  let rss = 0;
  for (let t = start; t < z.length; t++) {
    let pred = coef[0]!;
    for (let j = 1; j <= p; j++) pred += coef[j]! * z[t - j]!;
    const e = z[t]! - pred;
    residuals.push(e);
    rss += e * e;
  }
  return { coef, residuals, rss };
}

function aicc(rss: number, rows: number, params: number): number {
  const s2 = Math.max(rss / rows, 1e-12);
  const denom = rows - params - 1;
  if (denom <= 0) return Number.POSITIVE_INFINITY;
  return rows * Math.log(s2) + 2 * params + (2 * params * (params + 1)) / denom;
}

function arForecast(
  z: readonly number[],
  coef: readonly number[],
  h: number,
): number[] {
  const p = coef.length - 1;
  const hist = z.slice(-p);
  const out: number[] = [];
  for (let k = 0; k < h; k++) {
    let pred = coef[0]!;
    for (let j = 1; j <= p; j++) pred += coef[j]! * hist[hist.length - j]!;
    out.push(pred);
    hist.push(pred);
  }
  return out;
}

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ============================ Simple exponential smoothing ============================ */

const ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9];

function bestSes(y: readonly number[]) {
  let best: {
    sse: number;
    alpha: number;
    level: number;
    res: number[];
  } | null = null;
  for (const alpha of ALPHAS) {
    let l = y[0]!;
    let sse = 0;
    const res: number[] = [];
    for (let t = 1; t < y.length; t++) {
      const e = y[t]! - l;
      sse += e * e;
      res.push(e);
      l += alpha * e;
    }
    if (!best || sse < best.sse) best = { sse, alpha, level: l, res };
  }
  return best!;
}

const ses: ExtraModel = {
  id: "ses",
  label: "Simple exponential smoothing",
  needsSeason: false,
  fit: (y) => {
    if (y.length < 4) return null;
    const { level, res } = bestSes(y);
    return { residuals: res, forecast: (h) => new Array(h).fill(level) };
  },
};

/* ============================ Theta method ============================ */

const theta: ExtraModel = {
  id: "theta",
  label: "Theta method (smoothing + half the trend)",
  needsSeason: false,
  fit: (y) => {
    const n = y.length;
    if (n < 6) return null;
    const { alpha, level, res } = bestSes(y);
    const { b } = olsLine(y);
    return {
      residuals: res,
      forecast: (h) =>
        Array.from({ length: h }, (_, i) => {
          const k = i + 1;
          const drift =
            0.5 * b * (k - 1 + 1 / alpha - (1 - alpha) ** n / alpha);
          return level + drift;
        }),
    };
  },
};

/* ============================ Quadratic regression ============================ */

const quadratic: ExtraModel = {
  id: "quadratic",
  label: "Quadratic regression (curved trend)",
  needsSeason: false,
  fit: (y) => {
    const n = y.length;
    if (n < 8) return null;
    // Scale time to [0, 1] so the normal equations stay well conditioned.
    const s = (t: number) => t / (n - 1);
    const XtX = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    const Xty = [0, 0, 0];
    for (let t = 0; t < n; t++) {
      const v = [1, s(t), s(t) ** 2];
      for (let a = 0; a < 3; a++) {
        Xty[a]! += v[a]! * y[t]!;
        for (let b = 0; b < 3; b++) XtX[a]![b]! += v[a]! * v[b]!;
      }
    }
    const c = solve(XtX, Xty);
    if (!c) return null;
    const at = (t: number) => c[0]! + c[1]! * s(t) + c[2]! * s(t) ** 2;
    const lo = Math.min(...y);
    const hi = Math.max(...y);
    const range = Math.max(hi - lo, 1e-9);
    return {
      residuals: y.map((v, t) => v - at(t)),
      // A parabola explodes far out; keep it within one range of the data.
      forecast: (h) =>
        Array.from({ length: h }, (_, k) =>
          Math.min(hi + range, Math.max(lo - range, at(n + k))),
        ),
    };
  },
};

/* ============================ ARIMA(p, d, 0) ============================ */

const MAX_P = 5;

const arima: ExtraModel = {
  id: "arima",
  label: "ARIMA (autoregression on the series or its changes)",
  needsSeason: false,
  fit: (y) => {
    const n = y.length;
    if (n < 24) return null;
    const diff = y.slice(1).map((v, i) => v - y[i]!);
    // Every candidate is scored on the same rows so AICc is comparable.
    const start = MAX_P + 1;
    let best: {
      d: 0 | 1;
      p: number;
      coef: number[];
      residuals: number[];
      score: number;
    } | null = null;
    for (const d of [0, 1] as const) {
      const z = d === 0 ? y : diff;
      const offset = d === 0 ? 0 : 1; // diff[i] is y[i + 1] − y[i]
      const s0 = start - offset;
      for (let p = 1; p <= MAX_P; p++) {
        const f = fitAr(z, p, s0);
        if (!f) continue;
        const score = aicc(f.rss, f.residuals.length, p + 1);
        if (!best || score < best.score)
          best = { d, p, coef: f.coef, residuals: f.residuals, score };
      }
    }
    if (!best) return null;
    const { d, coef, residuals } = best;
    return {
      residuals,
      forecast: (h) => {
        if (d === 0) return arForecast(y, coef, h);
        const dz = arForecast(diff, coef, h);
        const out: number[] = [];
        let level = y[n - 1]!;
        for (const v of dz) {
          level += v;
          out.push(level);
        }
        return out;
      },
    };
  },
};

/* ============================ SARIMA(p,0,0)(0,1,0)m ============================ */

const sarima: ExtraModel = {
  id: "sarima",
  label: "SARIMA (seasonal differencing + autoregression)",
  needsSeason: true,
  fit: (y, m) => {
    const n = y.length;
    if (m < 2 || n < 2 * m + 10) return null;
    const w = y.slice(m).map((v, i) => v - y[i]!);
    let best: {
      coef: number[];
      residuals: number[];
      score: number;
    } | null = null;
    const start = 5;
    for (let p = 1; p <= 4; p++) {
      const f = fitAr(w, p, start);
      if (!f) continue;
      const score = aicc(f.rss, f.residuals.length, p + 1);
      if (!best || score < best.score)
        best = { coef: f.coef, residuals: f.residuals, score };
    }
    if (!best) return null;
    const { coef, residuals } = best;
    return {
      residuals,
      forecast: (h) => {
        const dw = arForecast(w, coef, h);
        const ext = y.slice();
        for (let k = 0; k < h; k++) ext.push(ext[ext.length - m]! + dw[k]!);
        return ext.slice(n);
      },
    };
  },
};

/* ============================ Neural network ============================ */

const NN_HIDDEN = 8;
const NN_EPOCHS = 120;
const NN_MAX_ROWS = 600;

const neural: ExtraModel = {
  id: "neural",
  label: "Neural network (lagged values + seasonal phase)",
  needsSeason: false,
  fit: (y, m) => {
    const n = y.length;
    const w = Math.max(6, Math.min(24, m >= 2 ? Math.min(m, 24) : 12));
    if (n < w + 16) return null;

    // Remove the straight-line trend, then standardise, so the network only
    // has to learn the shape around the trend.
    const { a, b } = olsLine(y);
    const detr = y.map((v, t) => v - (a + b * t));
    const mu = mean(detr);
    const sd = Math.sqrt(mean(detr.map((v) => (v - mu) ** 2))) || 1;
    const z = detr.map((v) => (v - mu) / sd);

    const phase = (t: number): [number, number] =>
      m >= 2
        ? [Math.sin((2 * Math.PI * t) / m), Math.cos((2 * Math.PI * t) / m)]
        : [0, 0];
    const D = w + 2;
    const features = (hist: readonly number[], t: number): number[] => {
      const f = new Array<number>(D);
      for (let j = 0; j < w; j++) f[j] = hist[hist.length - 1 - j]!;
      const [s, c] = phase(t);
      f[w] = s;
      f[w + 1] = c;
      return f;
    };

    const first = Math.max(w, n - NN_MAX_ROWS);
    const X: number[][] = [];
    const T: number[] = [];
    for (let t = first; t < n; t++) {
      X.push(features(z.slice(t - w, t), t));
      T.push(z[t]!);
    }
    const rows = X.length;

    // Seeded initialisation: the same data always gives the same network.
    const rand = mulberry32(20261123);
    const H = NN_HIDDEN;
    const init = (scale: number) => (rand() * 2 - 1) * scale;
    const W1 = Array.from({ length: H }, () =>
      Array.from({ length: D }, () => init(Math.sqrt(1 / D))),
    );
    const b1 = new Array<number>(H).fill(0);
    const W2 = Array.from({ length: H }, () => init(Math.sqrt(1 / H)));
    let b2 = 0;

    // Adam, full batch.
    const params = H * D + H + H + 1;
    const mAdam = new Array<number>(params).fill(0);
    const vAdam = new Array<number>(params).fill(0);
    const lr = 0.02;
    const decay = 1e-4;
    const hidden = new Array<number>(H).fill(0);
    for (let epoch = 1; epoch <= NN_EPOCHS; epoch++) {
      const gW1 = Array.from({ length: H }, () => new Array<number>(D).fill(0));
      const gb1 = new Array<number>(H).fill(0);
      const gW2 = new Array<number>(H).fill(0);
      let gb2 = 0;
      for (let r = 0; r < rows; r++) {
        const x = X[r]!;
        let out = b2;
        for (let h = 0; h < H; h++) {
          let s = b1[h]!;
          const wh = W1[h]!;
          for (let j = 0; j < D; j++) s += wh[j]! * x[j]!;
          hidden[h] = Math.tanh(s);
          out += W2[h]! * hidden[h]!;
        }
        const err = (2 * (out - T[r]!)) / rows;
        gb2 += err;
        for (let h = 0; h < H; h++) {
          gW2[h]! += err * hidden[h]!;
          const dh = err * W2[h]! * (1 - hidden[h]! ** 2);
          gb1[h]! += dh;
          const gw = gW1[h]!;
          for (let j = 0; j < D; j++) gw[j]! += dh * x[j]!;
        }
      }
      let idx = 0;
      const step = (value: number, grad: number) => {
        const g = grad + decay * value;
        mAdam[idx] = 0.9 * mAdam[idx]! + 0.1 * g;
        vAdam[idx] = 0.999 * vAdam[idx]! + 0.001 * g * g;
        const mh = mAdam[idx]! / (1 - 0.9 ** epoch);
        const vh = vAdam[idx]! / (1 - 0.999 ** epoch);
        idx++;
        return value - (lr * mh) / (Math.sqrt(vh) + 1e-8);
      };
      for (let h = 0; h < H; h++)
        for (let j = 0; j < D; j++) W1[h]![j] = step(W1[h]![j]!, gW1[h]![j]!);
      for (let h = 0; h < H; h++) b1[h] = step(b1[h]!, gb1[h]!);
      for (let h = 0; h < H; h++) W2[h] = step(W2[h]!, gW2[h]!);
      b2 = step(b2, gb2);
    }

    const predict = (x: readonly number[]) => {
      let out = b2;
      for (let h = 0; h < H; h++) {
        let s = b1[h]!;
        for (let j = 0; j < D; j++) s += W1[h]![j]! * x[j]!;
        out += W2[h]! * Math.tanh(s);
      }
      return out;
    };

    const residuals: number[] = [];
    for (let r = 0; r < rows; r++) {
      residuals.push((T[r]! - predict(X[r]!)) * sd);
    }
    if (residuals.some((e) => !Number.isFinite(e))) return null;

    return {
      residuals,
      forecast: (h) => {
        const hist = z.slice();
        const out: number[] = [];
        for (let k = 0; k < h; k++) {
          const t = n + k;
          let p = predict(features(hist, t));
          // Keep recursive predictions inside the range seen in training.
          p = Math.max(-4, Math.min(4, p));
          hist.push(p);
          out.push(p * sd + mu + a + b * t);
        }
        return out;
      },
    };
  },
};

export const EXTRA_MODELS: ExtraModel[] = [
  ses,
  theta,
  quadratic,
  arima,
  sarima,
  neural,
];
