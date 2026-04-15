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
  const body: RecommendRequest = await req.json();
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL;

  try {
    if (!backendUrl) throw new Error("NEXT_PUBLIC_BACKEND_URL not configured");
    const upstream = await fetch(backendUrl + "/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
    });

    if (!upstream.ok) {
      throw new Error("Backend returned " + upstream.status);
    }

    const payload = (await upstream.json()) as RecommendResponse;
    return NextResponse.json(payload);
  } catch {
    const fallback = fallbackForRequest(body);
    return NextResponse.json(fallback, {
      headers: { "x-vibescents-fallback": "locked" },
    });
  }
}
