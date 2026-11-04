import { describe, expect, it } from "vitest";
import { normalizePrediction, recommendedBins, UNCERTAIN_THRESHOLD, imageSchema } from "@/lib/scan-schema";
import { pilot, rowToScan, stats, type Scan } from "@/lib/eco-store";

const base = { item: "Bottle", reasoning: "PET bottle.", steps: ["Rinse"] };

describe("classification normalization", () => {
  it("maps each category to its server-owned bin", () => {
    for (const c of ["organic", "recyclable", "hazardous", "ewaste", "residual"] as const)
      expect(normalizePrediction({ ...base, category: c, confidence: 90 }, "gateway").bin).toBe(recommendedBins[c]);
  });
  it("marks predictions below the confidence threshold as uncertain", () => {
    const r = normalizePrediction({ ...base, category: "recyclable", confidence: UNCERTAIN_THRESHOLD - 1 }, "python");
    expect(r.category).toBe("uncertain");
    expect(r.bin).toBe(recommendedBins.uncertain);
  });
  it("keeps predictions at the threshold", () => {
    expect(normalizePrediction({ ...base, category: "organic", confidence: UNCERTAIN_THRESHOLD }, "gateway").category).toBe("organic");
  });
  it("rejects malformed model output instead of inventing a result", () => {
    expect(() => normalizePrediction({ item: "x", category: "plastic", confidence: 80, reasoning: "r" }, "gateway")).toThrow();
  });
  it("rejects non-image uploads", () => {
    expect(imageSchema.safeParse("data:text/plain;base64,aGVsbG8=").success).toBe(false);
  });
  it("rejects images over the character limit", () => {
    const oversized = `data:image/jpeg;base64,${"A".repeat(8_000_000)}`;
    expect(oversized.length).toBeGreaterThan(8_000_000);
    expect(imageSchema.safeParse(oversized).success).toBe(false);
  });
});

describe("dashboard stats", () => {
  const mk = (category: Scan["category"], confidence: number) => ({ category, confidence } as Scan);
  it("counts hazardous and e-waste separately, and averages only confident scans", () => {
    const s = stats([mk("organic", 80), mk("ewaste", 60), mk("hazardous", 70), mk("uncertain", 20)]);
    expect(s).toMatchObject({ total: 4, organic: 1, hazardous: 1, ewaste: 1, uncertain: 1, avgConfidence: 70 });
  });
  it("drops rows with unknown categories", () => {
    expect(rowToScan({ id: "1", item: "x", category: "glass", bin: "b", confidence: 1, reasoning: "r", steps: [], image: "", source: "gateway", created_at: new Date().toISOString() })).toBeNull();
  });
});

describe("pilot verification", () => {
  const v = (category: Scan["category"], verifiedCategory: Scan["verifiedCategory"]) => ({ category, verifiedCategory } as Scan);
  it("computes accuracy only from human-verified scans, ignoring unsure verdicts", () => {
    const p = pilot([v("organic", "organic"), v("recyclable", "residual"), v("ewaste", "unsure"), v("organic", null)]);
    expect(p).toEqual({ verified: 2, verifiedCorrect: 1, pilotAccuracy: 50 });
  });
  it("counts an uncertain AI result as not matching", () => {
    expect(pilot([v("uncertain", "organic")]).pilotAccuracy).toBe(0);
  });
  it("reports no accuracy when nothing is verified", () => {
    expect(pilot([v("organic", null)]).pilotAccuracy).toBeNull();
  });
});

describe("python classifier auth", () => {
  it("sends the classifier key header server-side", async () => {
    const { viaPython } = await import("@/lib/classifier.server");
    let sent: Headers | undefined;
    const orig = globalThis.fetch;
    globalThis.fetch = (async (_u: string, init: RequestInit) => {
      sent = new Headers(init.headers);
      return new Response(JSON.stringify({ item: "peel", category: "organic", confidence: 90, reasoning: "r", steps: ["a"] }), { status: 200 });
    }) as typeof fetch;
    try {
      await viaPython("https://py.example", "data:image/png;base64,iVBORw0KGgo=", "k123");
    } finally { globalThis.fetch = orig; }
    expect(sent?.get("X-Classifier-Key")).toBe("k123");
  });
});
