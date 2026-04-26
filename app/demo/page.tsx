"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import OutfitUploader from "@/components/demo/OutfitUploader";
import ContextForm from "@/components/demo/ContextForm";
import SubmitButton from "@/components/demo/SubmitButton";
import ResultsPanel from "@/components/demo/ResultsPanel";
import { getRecommendations } from "@/lib/recommend";
import { ContextInput, FragranceRecommendation } from "@/lib/types";

export default function DemoPage() {
  const [image, setImage] = useState<string | null>(null);
  const [mimeType, setMimeType] = useState<string | null>(null);
  const [context, setContext] = useState<ContextInput>({});
  const [budget, setBudget] = useState<number>(150);
  const [results, setResults] = useState<FragranceRecommendation[] | null>(
    null,
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Triggered when the OutfitUploader successfully processes an image
  const handleImageReady = (base64: string, mime: string) => {
    setImage(base64);
    setMimeType(mime);
    setResults(null); // Clear previous results on new image upload
    setError(null);
  };

  // Triggered by the SubmitButton
  const handleSubmit = async () => {
    if (!image || !mimeType) return;

    setLoading(true);
    setError(null);

    try {
      const response = await getRecommendations({
        image,
        mimeType,
        context,
        budget,
      });
      setResults(response.recommendations);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "An unexpected error occurred.",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-6 md:px-12 pt-32 pb-24 min-h-screen">
      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
        className="mb-16"
      >
        <h1 className="font-serif text-4xl md:text-5xl text-ink mb-4">
          Curate Your Scent
        </h1>
        <p className="font-sans text-muted max-w-2xl text-lg">
          Upload your outfit and define the moment. Our vision model will
          analyze the textures, palette, and style to recommend your perfect
          olfactory match.
        </p>
      </motion.div>

      {/* Main Layout Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-24">
        {/* Left Column: Inputs */}
        <div className="lg:col-span-5 flex flex-col gap-12">
          <OutfitUploader onImageReady={handleImageReady} />

          <div className="h-px w-full bg-border" />

          <ContextForm
            value={context}
            onChange={setContext}
            budget={budget}
            onBudgetChange={setBudget}
          />

          <div className="pt-4">
            {error && (
              <p className="font-mono text-xs tracking-wide text-bg bg-gold px-3 py-2 rounded-sm mb-4 text-center">
                {error}
              </p>
            )}
            <SubmitButton
              onClick={handleSubmit}
              disabled={!image}
              loading={loading}
            />
          </div>
        </div>

        {/* Right Column: Results */}
        <div className="lg:col-span-7 flex">
          <div className="sticky top-32 w-full h-full">
            <ResultsPanel results={results} loading={loading} />
          </div>
        </div>
      </div>
    </div>
  );
}
