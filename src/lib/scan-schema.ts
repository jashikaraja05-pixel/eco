import { z } from "zod";

export const MAX_IMAGE_CHARS = 8_000_000;
/** Below this confidence a prediction is reported as "uncertain" instead of a bin. */
export const UNCERTAIN_THRESHOLD = 55;

export const imageSchema = z.string().max(MAX_IMAGE_CHARS, "Image is too large (max ~6 MB).").regex(/^data:image\/(jpeg|png|webp);base64,[A-Za-z0-9+/]+={0,2}$/, "Choose a valid JPEG, PNG or WebP image.").superRefine((image, ctx) => {
  try {
    const encoded = image.split(",")[1];
    if (!encoded) throw new Error();
    const bytes = atob(encoded.slice(0, 64));
    const jpeg = bytes.startsWith("\xff\xd8\xff");
    const png = bytes.startsWith("\x89PNG\r\n\x1a\n");
    const webp = bytes.startsWith("RIFF") && bytes.slice(8, 12) === "WEBP";
    if (!(image.startsWith("data:image/jpeg;") ? jpeg : image.startsWith("data:image/png;") ? png : webp)) throw new Error();
  } catch { ctx.addIssue({ code: z.ZodIssueCode.custom, message: "Image content does not match its format." }); }
});

export const wasteCategories = ["organic", "recyclable", "hazardous", "ewaste", "residual"] as const;
export type WasteCategory = (typeof wasteCategories)[number];
export type StoredCategory = WasteCategory | "uncertain";

export const recommendedBins: Record<StoredCategory, string> = {
  organic: "Green Bin (Organic / Wet Waste)",
  recyclable: "Blue Bin (Dry Recyclable Waste)",
  hazardous: "Red Bin (Hazardous Waste Depot)",
  ewaste: "Red Bin (Special E-Waste Bin)",
  residual: "Black Bin (Residual Non-Recyclable Waste)",
  uncertain: "No bin — rescan or check local rules",
};

/** Safety notes shown with every result of a category. Mirrors SAFETY_WARNINGS in backend/common.py. */
export const safetyWarnings: Record<StoredCategory, string[]> = {
  hazardous: [
    "Do not open, puncture, crush or burn batteries, aerosol cans or chemical containers.",
    "Keep away from children and food; tape battery terminals before drop-off.",
  ],
  ewaste: [
    "Do not dismantle the device or remove its battery yourself.",
    "Swollen or damaged batteries can catch fire — store them away from heat.",
  ],
  organic: ["Remove stickers, rubber bands and plastic before composting."],
  recyclable: ["Broken glass: wrap it in paper and label it before binning."],
  residual: [],
  uncertain: ["Bin colours differ between cities — confirm with your municipal corporation's guide."],
};

/** Reuse / recycling ideas per category (static guidance, not model output). */
export const reuseTips: Record<StoredCategory, string> = {
  organic: "Compost at home (a pot composter works in flats) or give to your society's composting unit.",
  recyclable: "Clean, dry dry-waste fetches value: hand it to your kabadiwala or a dry-waste collection centre.",
  hazardous: "Use your city's household hazardous-waste drive; many retailers accept used batteries.",
  ewaste: "Working devices can be donated; otherwise use a CPCB-authorised recycler or brand take-back.",
  residual: "Reduce next time — choose refills or unpackaged options where possible.",
  uncertain: "Rescan with one item, plain background and good light.",
};

/** Raw model/Python output shape (loosely validated, then normalized). */
export const rawPredictionSchema = z.object({
  item: z.string().trim().min(1).max(150),
  category: z.enum([...wasteCategories, "uncertain"]),
  confidence: z.coerce.number().min(0).max(100),
  reasoning: z.string().trim().min(1).max(1500),
  steps: z.array(z.string().trim().min(1).max(500)).max(6).default([]),
});

export type ScanResult = {
  item: string;
  category: StoredCategory;
  bin: string;
  confidence: number;
  reasoning: string;
  steps: string[];
  source: "gateway" | "python";
};

/** Server-owned bin mapping + uncertainty rule; never trusts the model's bin text. */
export function normalizePrediction(raw: unknown, source: ScanResult["source"]): ScanResult {
  const p = rawPredictionSchema.parse(raw);
  const confidence = Math.round(p.confidence);
  const uncertain = p.category === "uncertain" || confidence < UNCERTAIN_THRESHOLD;
  const category: StoredCategory = uncertain ? "uncertain" : p.category;
  const steps = uncertain
    ? ["Retake the photo in good light with only one item in frame.", "If still unsure, keep it aside and check your municipality's segregation guide."]
    : p.steps.length ? p.steps : ["Follow your local municipal guidance for this bin."];
  return { item: p.item, category, bin: recommendedBins[category], confidence, reasoning: p.reasoning, steps, source };
}
