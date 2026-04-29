import { NextRequest, NextResponse } from "next/server";
import type { RecommendRequest, RecommendResponse } from "@/lib/types";
import { z } from "zod";

const DEFAULT_BUDGET_USD = 150.0;
const ML_BACKEND_URL = process.env.ML_BACKEND_URL ?? "http://127.0.0.1:8000";
const SCRAPER_BACKEND_URL =
  process.env.SCRAPER_BACKEND_URL ?? "http://127.0.0.1:8001";

const RequestSchema = z.object({
  image: z.string().min(1).max(14_000_000),
  mimeType: z.enum(["image/jpeg", "image/png", "image/webp"]),
  context: z.object({
    eventType: z.string().optional(),
    timeOfDay: z.string().optional(),
    mood: z.string().optional(),
  }),
  budget: z.number().positive().max(10_000).optional(),
});

function contextToDescription(ctx: RecommendRequest["context"]): string {
  return [ctx.eventType, ctx.timeOfDay, ctx.mood].filter(Boolean).join(", ");
}

function hasText(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

export async function POST(req: NextRequest) {
  let body: RecommendRequest;
  const contentLength = parseInt(req.headers.get('content-length') ?? '0', 10);
  if (contentLength > 14 * 1024 * 1024) {
    return NextResponse.json({ error: 'Payload too large. Maximum image size is 10MB.' }, { status: 413 });
  }
  try {
    const parsed = RequestSchema.safeParse(await req.json());
    if (!parsed.success) {
      return NextResponse.json({ error: 'Invalid request', details: parsed.error.flatten() }, { status: 400 });
    }
    body = parsed.data as RecommendRequest;
    console.log(`\n--- New Recommendation Request ---`);
    console.log(`Context:`, body.context);
    console.log(`MimeType:`, body.mimeType);

    // Ensure the image string is pure base64 (strip data URI prefix if present)
    const base64Image = body.image.replace(/^data:image\/\w+;base64,/, "");
    console.log(`Base64 Image length:`, base64Image.length);

    // 1. Call the ML Model
    console.log(`Calling ML Backend at ${ML_BACKEND_URL}/recommend...`);
    const mlResponse = await fetch(`${ML_BACKEND_URL}/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: base64Image,
        mimeType: body.mimeType,
        context: body.context,
      }),
      signal: AbortSignal.timeout(120_000),  // 120s — embedder warm-up + NIM reranking takes time
    });

    if (!mlResponse.ok) {
      const errorText = await mlResponse.text();
      console.error(`ML Backend Error (${mlResponse.status}):`, errorText);
      throw new Error(`ML backend returned ${mlResponse.status}: ${errorText}`);
    }

    const mlResult = (await mlResponse.json()) as RecommendResponse;
    console.log(
      `ML Backend returned ${mlResult.recommendations?.length || 0} recommendations.`,
    );

    // Extract the names of the top perfumes
    const perfumeNames = mlResult.recommendations.map(
      (frag) => `${frag.house} ${frag.name}`,
    );

    // Get the requested budget or fallback to 150
    const budgetCap =
      body.budget && body.budget > 0 ? body.budget : DEFAULT_BUDGET_USD;

    // 2. Call the Scraper API
    console.log(
      `Calling Scraper API at ${SCRAPER_BACKEND_URL}/search with budget $${budgetCap}...`,
    );
    
    let scraperResult = null;
    try {
      const scraperResponse = await fetch(`${SCRAPER_BACKEND_URL}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          perfumes: perfumeNames,
          budget: budgetCap,
        }),
        signal: AbortSignal.timeout(60_000), // Increased to 60s for SerpAPI
      });

      if (!scraperResponse.ok) {
        const scraperErrorText = await scraperResponse.text();
        console.warn(
          `Scraper failed (${scraperResponse.status}): ${scraperErrorText}`,
        );
      } else {
        scraperResult = await scraperResponse.json();
        console.log(
          `Scraper returned data for ${scraperResult?.length || 0} items.`,
        );
      }
    } catch (scraperErr) {
      console.warn("Scraper fetch failed or timed out:", scraperErr);
      // We continue with ML results even if scraper times out
    }

    if (!scraperResult) {
      console.warn("Returning ML results without pricing.");
      return NextResponse.json(mlResult);
    }

    // 3. Merge scraper pricing data into ML results
    const finalRecommendations = mlResult.recommendations.map((frag, index) => {
      const scrapedData = scraperResult[index];
      const rawPrice = scrapedData?.price ?? frag.price;
      const purchaseUrl =
        scrapedData?.purchaseUrl ?? scrapedData?.url ?? frag.purchaseUrl ?? "#";
      const store = scrapedData?.store ?? frag.store ?? "Retailer unavailable";
      const thumbnail = scrapedData?.thumbnail ?? frag.thumbnail ?? null;
      const inBudget =
        scrapedData?.inBudget ?? scrapedData?.in_budget ?? frag.inBudget ?? false;

      return {
        ...frag,
        price:
          typeof rawPrice === "number"
            ? `$${rawPrice.toFixed(2)}`
            : hasText(rawPrice)
              ? rawPrice
              : "Price unavailable",
        purchaseUrl: hasText(purchaseUrl) ? purchaseUrl : "#",
        store: hasText(store) ? store : "Retailer unavailable",
        thumbnail: hasText(thumbnail) ? thumbnail : null,
        inBudget: Boolean(inBudget),
      };
    });

    console.log("Successfully returning merged recommendations.");
    return NextResponse.json({ recommendations: finalRecommendations });
  } catch (error) {
    console.error("API Route Error:", error);

    // Surface the real error so users/devs understand what went wrong
    let message = "An unexpected error occurred.";
    let hint = "";

    if (error instanceof Error) {
      if (error.name === "TimeoutError" || error.name === "AbortError") {
        message = "The ML backend took too long to respond (>120s).";
        hint = "This usually happens on first request while the embedder model warms up. Try again in a few seconds.";
      } else if (error.message.includes("ECONNREFUSED") || error.message.includes("fetch failed")) {
        message = "Cannot connect to the ML backend at " + ML_BACKEND_URL + ".";
        hint = "Make sure the backend is running: run `bun run dev` in your project folder.";
      } else {
        message = error.message;
      }

    }

    return NextResponse.json(
      { error: message, hint },
      { status: 503 },
    );
  }
}
