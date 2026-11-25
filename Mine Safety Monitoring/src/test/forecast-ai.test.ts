// @vitest-environment node
import { describe, expect, it } from "vitest";

import { runAnalysis } from "@/lib/ai/analyst";
import {
  buildFactsSheet,
  findDirectionContradictions,
  planFindings,
} from "@/lib/ai/insights";
import type { LocalInferenceProvider } from "@/lib/ai/provider";
import { MINING_SYSTEM_PROMPT } from "@/lib/ai/prompts";
import { buildDataset, buildProfile } from "@/lib/dataset";
import { buildMarkdownReport } from "@/lib/report";

const rows = ["timestamp,methane_pct,airflow_m3s,smoke_detected"];
for (let i = 0; i < 48; i++) {
  const t = new Date(Date.UTC(2026, 8, 1, i))
    .toISOString()
    .slice(0, 16)
    .replace("T", " ");
  rows.push(
    `${t},${(0.3 + i * 0.01 + ((i * 7) % 5) * 0.004).toFixed(3)},${(44 - i * 0.2).toFixed(2)}`,
  );
}
const dataset = buildDataset(rows.join("\n"), "hourly.csv");
const profile = buildProfile(dataset);

const provider = (answer: unknown): LocalInferenceProvider => ({
  name: "fake",
  model: "fake:1b",
  status: async () => {
    throw new Error("unused");
  },
  generateStructured: async () => ({
    value: answer,
    model: "fake:1b",
    durationMs: 1,
  }),
});

const answer = {
  headline: "Methane rose while airflow fell",
  summary: "Methane_pct rose over the period. Airflow_m3s fell.",
  notes: [],
  recommendations: ["Inspect the main fan."],
};

describe("forecast in the AI report", () => {
  it("adds a compact forecast to the profile", () => {
    expect(profile.forecast?.horizon).toBe(24);
    const m = profile.forecast!.items.find((i) => i.column === "methane_pct")!;
    expect(m.end).toBeGreaterThan(m.last);
    expect(m.lo).toBeLessThanOrEqual(m.end);
    expect(m.hi).toBeGreaterThanOrEqual(m.end);
  });

  it("does not forecast 0/1 detection flags", () => {
    expect(profile.forecast!.items.map((i) => i.column)).not.toContain(
      "smoke_detected",
    );
  });

  it("puts the forecast in the facts the local model reads", () => {
    const sheet = buildFactsSheet(profile);
    expect(sheet).toContain("FORECAST (statistical projection");
    expect(sheet).toMatch(/- methane_pct: expected to go up from/);
    expect(sheet).toMatch(/- airflow_m3s: expected to go down from/);
  });

  it("tells the model a forecast is an expectation that never changes the status", () => {
    expect(MINING_SYSTEM_PROMPT).toMatch(
      /FORECAST lines are statistical expectations/,
    );
    expect(MINING_SYSTEM_PROMPT).toMatch(/never a breach/);
  });

  it("writes the outlook with code and attaches it to the result", async () => {
    const res = await runAnalysis(profile, { provider: provider(answer) });
    expect(res.ok).toBe(true);
    if (!res.ok) return;
    const outlook = res.insights.outlook ?? [];
    expect(outlook.map((o) => o.parameter).sort()).toEqual([
      "airflow_m3s",
      "methane_pct",
    ]);
    expect(outlook.find((o) => o.parameter === "methane_pct")!.text).toMatch(
      /expected to rise about/,
    );
    const md = buildMarkdownReport(dataset, {
      insights: res.insights,
      model: "fake:1b",
      analysedAt: new Date().toISOString(),
    });
    expect(md).toContain("## Forecast outlook");
  });

  it("does not flag a forecast sentence as contradicting the recorded trend", () => {
    const planned = planFindings(profile);
    expect(planned.length).toBeGreaterThan(0);
    // airflow fell in the record; saying it is expected to rise is a forecast claim.
    expect(
      findDirectionContradictions(
        "Airflow_m3s is expected to rise next shift.",
        profile,
      ),
    ).toEqual([]);
    expect(
      findDirectionContradictions("Airflow_m3s rose over the period.", profile),
    ).toEqual(["airflow_m3s"]);
  });

  it("still works when a column is too short to forecast", () => {
    const short = buildProfile(
      buildDataset("t,x\n2026-01-01,1\n2026-01-02,2\n2026-01-03,3", "s.csv"),
    );
    expect(short.forecast).toBeUndefined();
  });
});
