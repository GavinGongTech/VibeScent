"use client";

import { FragranceRecommendation } from "@/lib/types";
import { motion } from "framer-motion";
import FragranceCard from "./FragranceCard";

interface ResultsPanelProps {
  results: FragranceRecommendation[] | null;
  loading: boolean;
}

export default function ResultsPanel({ results, loading }: ResultsPanelProps) {
  // Loading State (Custom minimal skeletons)
  if (loading) {
    return (
      <div className="flex flex-col gap-6 w-full">
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className="w-full h-96 border border-border bg-surface rounded-sm relative overflow-hidden"
          >
            {/* Elegant shimmer effect */}
            <motion.div
              animate={{ x: ["-100%", "200%"] }}
              transition={{ repeat: Infinity, duration: 1.5, ease: "linear" }}
              className="absolute inset-0 bg-linear-to-r from-transparent via-gold/5 to-transparent w-1/2"
            />
          </div>
        ))}
      </div>
    );
  }

  // Populated State
  if (results && results.length > 0) {
    return (
      <div className="flex flex-col gap-6 w-full">
        {results.map((fragrance, index) => (
          <FragranceCard
            key={fragrance.name}
            fragrance={fragrance}
            index={index}
          />
        ))}
      </div>
    );
  }

  // Placeholder State (Empty)
  return (
    <div className="w-full h-full border border-dashed border-border rounded-sm flex items-center justify-center bg-surface/30">
      <div className="text-center">
        <div className="w-8 h-px bg-gold/50 mx-auto mb-6" />
        <p className="font-serif text-2xl text-muted italic">
          Your curation will appear here.
        </p>
        <p className="font-sans text-sm tracking-widest text-border uppercase mt-4">
          Awaiting outfit data
        </p>
      </div>
    </div>
  );
}
