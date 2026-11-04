import { normalizePrediction, type ScanResult } from "./scan-schema";

const SYSTEM = `You are a waste segregation expert for Indian municipal rules. Look at the image and classify the single main waste item.
Categories:
- organic: food scraps, peels, natural/garden waste
- recyclable: clean plastic bottles, paper, cardboard, glass, metal cans
- hazardous: aerosol cans, chemicals, paint, batteries, medicines
- ewaste: electronics, circuit boards, cables, phones
- residual: multi-layer packaging, chip/biscuit wrappers, laminates, soiled tissues
- uncertain: the image is blurry, contains no waste item, several mixed items, or you cannot tell
confidence is your honest probability (0-100) that the category is correct.
Reply with ONLY a JSON object: {"item": string, "category": one of the keys above, "confidence": number, "reasoning": string (max 2 sentences), "steps": string[] (3-5 disposal steps)}`;

export async function viaPython(url: string, image: string, apiKey?: string): Promise<ScanResult> {
  const [meta, b64] = image.split(",");
  const mime = meta!.slice(5, meta!.indexOf(";"));
  const bin = Uint8Array.from(atob(b64!), (c) => c.charCodeAt(0));
  const form = new FormData();
  form.append("file", new Blob([bin], { type: mime }), `scan.${mime.split("/")[1]}`);
  let res: Response;
  try { res = await fetch(`${url.replace(/\/$/, "")}/classify`, { method: "POST", body: form, headers: apiKey ? { "X-Classifier-Key": apiKey } : {} }); }
  catch { throw new Error("Python classifier service is unreachable."); }
  const body = (await res.json().catch(() => null)) as { detail?: string } | null;
  if (!res.ok) throw new Error(body?.detail || `Classifier service error (${res.status}).`);
  return normalizePrediction(body, "python");
}

export async function viaGateway(apiKey: string, image: string): Promise<ScanResult> {
  const { createOpenAI } = await import("@ai-sdk/openai");
  const { streamText } = await import("ai");
  const provider = createOpenAI({
    baseURL: "https://ai.gateway.lovable.dev/v1",
    apiKey,
    headers: { "Lovable-API-Key": apiKey, "X-Lovable-AIG-SDK": "vercel-ai-sdk" },
  });
  const result = streamText({
    model: provider.responses("openai/gpt-6-astra"),
    system: SYSTEM,
    messages: [{ role: "user", content: [{ type: "text", text: "Classify this waste item. Respond in json." }, { type: "file", mediaType: image.slice(5, image.indexOf(";")), data: new URL(image) }] }],
    providerOptions: { openai: { store: false, forceReasoning: true, reasoningEffort: "low", include: ["reasoning.encrypted_content"] } },
  });
  const text = await result.text;
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) throw new Error("The AI could not analyze this image. Try a clearer photo.");
  return normalizePrediction(JSON.parse(match[0]), "gateway");
}

/**
 * Routes to the Python service when its URL is configured, otherwise the built-in gateway.
 * No fallback: a Python failure is surfaced to the user, never silently replaced.
 */
export async function classifyImage(image: string, env: { pythonUrl?: string | undefined; pythonApiKey?: string | undefined; lovableApiKey?: string | undefined }): Promise<ScanResult> {
  if (env.pythonUrl) return viaPython(env.pythonUrl, image, env.pythonApiKey);
  if (env.lovableApiKey) return viaGateway(env.lovableApiKey, image);
  throw new Error("AI classification is not configured.");
}
