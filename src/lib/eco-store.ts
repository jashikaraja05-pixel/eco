import type { ScanResult, StoredCategory } from "./scan-schema";

export type VerifiedCategory = Exclude<StoredCategory, "uncertain"> | "unsure";
export type Scan = ScanResult & { id: string; image: string; createdAt: number; verifiedCategory: VerifiedCategory | null; feedback: string | null };

export type ScanRow = {
  id: string; item: string; category: string; bin: string; confidence: number | null;
  reasoning: string; steps: unknown; image: string; source: string; created_at: string;
  verified_category?: string | null; feedback?: string | null;
};

const verifiedCats: VerifiedCategory[] = ["organic", "recyclable", "hazardous", "ewaste", "residual", "unsure"];

const cats: StoredCategory[] = ["organic", "recyclable", "hazardous", "ewaste", "residual", "uncertain"];

/** Converts a database row into a UI scan, dropping malformed rows instead of crashing. */
export function rowToScan(r: ScanRow): Scan | null {
  if (!cats.includes(r.category as StoredCategory)) return null;
  const steps = Array.isArray(r.steps) ? r.steps.filter((s): s is string => typeof s === "string") : [];
  return {
    id: r.id, item: r.item, category: r.category as StoredCategory, bin: r.bin,
    confidence: r.confidence ?? 0, reasoning: r.reasoning, steps, image: r.image,
    source: r.source === "python" ? "python" : "gateway", createdAt: new Date(r.created_at).getTime(),
    verifiedCategory: verifiedCats.includes(r.verified_category as VerifiedCategory) ? (r.verified_category as VerifiedCategory) : null,
    feedback: r.feedback ?? null,
  };
}

export const BINS = {
  organic: { label: "Organic", bin: "Green Bin", tone: "bg-bin-green" },
  recyclable: { label: "Recyclable", bin: "Blue Bin", tone: "bg-bin-blue" },
  hazardous: { label: "Hazardous", bin: "Red Bin", tone: "bg-bin-red" },
  ewaste: { label: "E-Waste", bin: "Red Bin", tone: "bg-bin-red" },
  residual: { label: "Residual", bin: "Black Bin", tone: "bg-bin-black" },
  uncertain: { label: "Uncertain", bin: "Rescan", tone: "bg-muted-foreground" },
} as const;

export function stats(list: Scan[]) {
  const c = (k: string[]) => list.filter((s) => k.includes(s.category)).length;
  const confident = list.filter((s) => s.category !== "uncertain");
  return {
    total: list.length,
    organic: c(["organic"]),
    recyclable: c(["recyclable"]),
    hazardous: c(["hazardous"]),
    ewaste: c(["ewaste"]),
    residual: c(["residual"]),
    uncertain: c(["uncertain"]),
    ...pilot(list),
    avgConfidence: confident.length ? Math.round(confident.reduce((a, s) => a + s.confidence, 0) / confident.length) : 0,
  };
}

/**
 * Pilot results from human-verified scans only. "unsure" verdicts are excluded
 * from accuracy. An uncertain AI result counts as not matching (no bin given).
 */
export function pilot(list: Scan[]) {
  const judged = list.filter((s) => s.verifiedCategory && s.verifiedCategory !== "unsure");
  const correct = judged.filter((s) => s.category === s.verifiedCategory).length;
  return {
    verified: judged.length,
    verifiedCorrect: correct,
    pilotAccuracy: judged.length ? Math.round((correct / judged.length) * 100) : null,
  };
}
