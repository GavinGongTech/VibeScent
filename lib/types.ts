// Input sent to the model
export interface RecommendRequest {
  image: string; // base64-encoded image data
  mimeType: string; // "image/jpeg" | "image/png" | "image/webp"
  context: ContextInput;
}

export interface ContextInput {
  eventType?: string;
  timeOfDay?: string;
  mood?: string;
  customNotes?: string;
}

// Output returned by the model
export interface RecommendResponse {
  recommendations: FragranceRecommendation[];
}

export interface FragranceRecommendation {
  rank: number;
  name: string;
  house: string;
  score: number;
  notes: string[];
  reasoning: string;
  occasion: string;
  price?: string;
  purchaseUrl?: string;
  store?: string;
  thumbnail?: string | null;
  inBudget?: boolean;
}
