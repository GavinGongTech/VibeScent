import { RecommendRequest, RecommendResponse } from "./types";

export async function getRecommendations(
  payload: RecommendRequest,
  signal?: AbortSignal,
): Promise<RecommendResponse> {
  const response = await fetch("/api/recommend", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as {
      error?: string;
    };
    throw new Error(
      body.error ?? "Failed to curate fragrance. Please try again.",
    );
  }

  return response.json();
}
