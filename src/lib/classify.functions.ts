import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";
import { imageSchema, type ScanResult } from "./scan-schema";
import { classifyImage } from "./classifier.server";
import { rowToScan, type Scan, type ScanRow } from "./eco-store";

/** Classifies an image (Python FastAPI service if configured, else built-in AI) and saves it to the caller's history. */
export const classifyWaste = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((d) => z.object({ image: imageSchema }).parse(d))
  .handler(async ({ data, context }): Promise<Scan> => {
    let result: ScanResult;
    try {
      result = await classifyImage(data.image, { pythonUrl: process.env["PYTHON_CLASSIFIER_URL"], pythonApiKey: process.env["PYTHON_CLASSIFIER_API_KEY"], lovableApiKey: process.env["LOVABLE_API_KEY"] });
    } catch (e) {
      console.error("classification failed", e);
      throw new Error(e instanceof z.ZodError ? "The AI returned an invalid answer. Please try again." : (e as Error).message);
    }
    const { data: row, error } = await context.supabase
      .from("scans")
      .insert({ ...result, steps: result.steps, image: data.image, user_id: context.userId })
      .select()
      .single();
    if (error || !row) throw new Error("Classified, but could not save to your history.");
    return rowToScan(row as ScanRow)!;
  });

/** Permanently deletes the signed-in user's scans and account. */
export const deleteMyAccount = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }) => {
    await context.supabase.from("scans").delete().eq("user_id", context.userId);
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    const { error } = await supabaseAdmin.auth.admin.deleteUser(context.userId);
    if (error) throw new Error("Could not delete account.");
    return { ok: true };
  });
