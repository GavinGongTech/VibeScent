import { NextRequest, NextResponse } from "next/server";
import type { RecommendRequest, RecommendResponse } from "@/lib/types";

export async function POST(req: NextRequest) {
  const body: RecommendRequest = await req.json();

  // ── STUB ──────────────────────────────────────────────────────────────
  // Replace this block with your real model call later.
  // The body contains: body.image (base64), body.mimeType, body.context
  // Return a RecommendResponse-shaped object.
  // ──────────────────────────────────────────────────────────────────────

  // Simulate model processing time (1.5 seconds) so we can see our loading animation
  await new Promise((resolve) => setTimeout(resolve, 1500));

  const mockResponse: RecommendResponse = {
    recommendations: [
      {
        rank: 1,
        name: "Baccarat Rouge 540",
        house: "Maison Francis Kurkdjian",
        score: 0.91,
        notes: ["jasmine", "saffron", "amberwood", "fir resin"],
        reasoning:
          "The structured elegance of the outfit calls for a warm, luminous signature. Baccarat Rouge 540's crystalline amber and saffron accord mirrors the refined opulence of your look.",
        occasion: "Formal evening event",
      },
      {
        rank: 2,
        name: "Black Orchid",
        house: "Tom Ford",
        score: 0.84,
        notes: ["black truffle", "ylang ylang", "dark chocolate", "patchouli"],
        reasoning:
          "The deep tones in your palette align with this fragrance's dark floral complexity — a bold, commanding presence for a high-stakes occasion.",
        occasion: "Cocktail or gala",
      },
      {
        rank: 3,
        name: "Portrait of a Lady",
        house: "Frédéric Malle",
        score: 0.76,
        notes: ["turkish rose", "raspberry", "patchouli", "incense"],
        reasoning:
          "For a softer interpretation of the look, this rose-centered oriental brings warmth and depth without overpowering the visual statement you're making.",
        occasion: "Evening dinner",
      },
    ],
  };

  return NextResponse.json(mockResponse);
}
