import { describe, expect, it } from "vitest";
import { normalizePrediction, recommendedBins, UNCERTAIN_THRESHOLD, imageSchema } from "@/lib/scan-schema";
import { rowToScan, stats, type Scan } from "@/lib/eco-store";

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
  it("counts categories, grouping e-waste under hazardous, and averages only confident scans", () => {
    const s = stats([mk("organic", 80), mk("ewaste", 60), mk("hazardous", 70), mk("uncertain", 20)]);
    expect(s).toMatchObject({ total: 4, organic: 1, hazardous: 2, uncertain: 1, avgConfidence: 70 });
  });
  it("drops rows with unknown categories", () => {
    expect(rowToScan({ id: "1", item: "x", category: "glass", bin: "b", confidence: 1, reasoning: "r", steps: [], image: "", source: "gateway", created_at: new Date().toISOString() })).toBeNull();
  });
});
