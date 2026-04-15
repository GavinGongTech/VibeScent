import { RecommendRequest, RecommendResponse } from "./types";

export async function getRecommendations(
  payload: RecommendRequest,
): Promise<RecommendResponse> {
  const response = await fetch("/api/recommend", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error("Failed to curate fragrance. Please try again.");
  }

  return response.json();
}
