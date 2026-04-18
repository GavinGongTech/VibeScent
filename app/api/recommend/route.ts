import { NextRequest, NextResponse } from "next/server";
import type { RecommendRequest, RecommendResponse } from "@/lib/types";

const FALLBACK_RESPONSES: Record<string, RecommendResponse> = {
  default: {
    recommendations: [
      {
        rank: 1,
        name: "Baccarat Rouge 540",
        house: "Maison Francis Kurkdjian",
        score: 0.91,
        notes: ["jasmine", "saffron", "amberwood", "fir resin"],
        reasoning:
          "Fallback response: luminous amber profile matching formal evening styling.",
        occasion: "Formal evening event",
      },
      {
        rank: 2,
        name: "Black Orchid",
        house: "Tom Ford",
        score: 0.84,
        notes: ["black truffle", "ylang ylang", "dark chocolate", "patchouli"],
        reasoning:
          "Fallback response: darker floral profile for rich palettes and high contrast looks.",
        occasion: "Cocktail or gala",
      },
      {
        rank: 3,
        name: "Portrait of a Lady",
        house: "Frédéric Malle",
        score: 0.76,
        notes: ["turkish rose", "raspberry", "patchouli", "incense"],
        reasoning:
          "Fallback response: warm rose-forward option for evening dinner settings.",
        occasion: "Evening dinner",
      },
    ],
  },
};

function fallbackForRequest(_body: RecommendRequest): RecommendResponse {
  return FALLBACK_RESPONSES.default;
}

export async function POST(req: NextRequest) {
  try {
    const body = (await req.json()) as RecommendRequest;

    // 1. Call the ML Model (Port 8000)
    const mlResponse = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!mlResponse.ok) throw new Error("ML backend failed");
    const mlResult = (await mlResponse.json()) as RecommendResponse;

    // Extract the names of the top 3 perfumes
    const perfumeNames = mlResult.recommendations.map(
      (frag) => `${frag.house} ${frag.name}`,
    );

    // 2. Call the Scraper API (Port 8001)
    // We pass a default budget of 500, but you could eventually add this to your UI!
    const scraperResponse = await fetch("http://127.0.0.1:8001/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        perfumes: perfumeNames,
        budget: 500.0,
      }),
    });

    if (!scraperResponse.ok) {
      console.warn("Scraper failed, but returning ML results anyway.");
      return NextResponse.json(mlResult); // Fail gracefully!
    }

    const scraperResult = await scraperResponse.json();

    // 3. Merge the Scraper data into the ML data
    // (This assumes your scraper returns an array or dictionary we can match by name)
    const finalRecommendations = mlResult.recommendations.map((frag, index) => {
      // Scraper result could be the dictionary, or null if nothing was found
      const scrapedData = scraperResult[index];

      return {
        ...frag,
        // Format the float price to a beautiful string, fallback if null
        price: scrapedData?.price
          ? `$${scrapedData.price.toFixed(2)}`
          : "Price unavailable",

        // Map the exact keys from your python dictionary
        purchaseUrl: scrapedData?.url || "#",
        store: scrapedData?.store || "Retailer unavailable",
        thumbnail: scrapedData?.thumbnail || null,
      };
    });

    // 4. Send the complete package to the frontend
    return NextResponse.json({ recommendations: finalRecommendations });
  } catch (error) {
    console.error("API Route Error:", error);
    return NextResponse.json(
      { error: "Failed to generate recommendation" },
      { status: 500 },
    );
  }
}
