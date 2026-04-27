import { NextRequest, NextResponse } from "next/server";
import type { RecommendRequest, RecommendResponse } from "@/lib/types";

const DEFAULT_BUDGET_USD = 500.0;
const ML_BACKEND_URL = process.env.ML_BACKEND_URL ?? "http://127.0.0.1:8000";
const SCRAPER_BACKEND_URL =
  process.env.SCRAPER_BACKEND_URL ?? "http://127.0.0.1:8001";

function contextToDescription(ctx: RecommendRequest["context"]): string {
  return [ctx.eventType, ctx.timeOfDay, ctx.mood].filter(Boolean).join(", ");
}

export async function POST(req: NextRequest) {
  let body: RecommendRequest;
  try {
    body = (await req.json()) as RecommendRequest;
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
    const budgetCap = body.budget && body.budget > 0 ? body.budget : 150.0;

    // 2. Call the Scraper API
    console.log(
      `Calling Scraper API at ${SCRAPER_BACKEND_URL}/search with budget $${budgetCap}...`,
    );
    const scraperResponse = await fetch(`${SCRAPER_BACKEND_URL}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        perfumes: perfumeNames,
        budget: budgetCap,
      }),
    });

    if (!scraperResponse.ok) {
      const scraperErrorText = await scraperResponse.text();
      console.warn(
        `Scraper failed (${scraperResponse.status}): ${scraperErrorText}`,
      );
      console.warn("Returning ML results without pricing.");
      return NextResponse.json(mlResult);
    }

    const scraperResult = await scraperResponse.json();
    console.log(
      `Scraper returned data for ${scraperResult?.length || 0} items.`,
    );

    // 3. Merge scraper pricing data into ML results
    const finalRecommendations = mlResult.recommendations.map((frag, index) => {
      const scrapedData = scraperResult[index];
      return {
        ...frag,
        price:
          typeof scrapedData?.price === "number"
            ? `$${scrapedData.price.toFixed(2)}`
            : scrapedData?.price || "Price unavailable",
        purchaseUrl: scrapedData?.url || "#",
        store: scrapedData?.store || "Retailer unavailable",
        thumbnail: scrapedData?.thumbnail || null,
        inBudget: scrapedData?.in_budget ?? false,
      };
    });

    console.log("Successfully returning merged recommendations.");
    return NextResponse.json({ recommendations: finalRecommendations });
  } catch (error) {
    console.error("API Route Error:", error);
    return NextResponse.json(
      {
        error:
          "Model backend unavailable or encountered an error. Please ensure the ML server is running and check the logs.",
      },
      { status: 503 },
    );
  }
}
