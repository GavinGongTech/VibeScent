"use client";

import { FragranceRecommendation } from "@/lib/types";
import { motion } from "framer-motion";

interface FragranceCardProps {
  fragrance: FragranceRecommendation;
  index: number;
}

export default function FragranceCard({
  fragrance,
  index,
}: FragranceCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.8,
        ease: [0.16, 1, 0.3, 1],
        delay: index * 0.08, // Staggering the cards
      }}
      className="relative flex flex-col p-8 border border-border bg-surface group hover:border-gold/30 transition-colors duration-500 rounded-sm overflow-hidden"
    >
      {/* Subtle hover overlay */}
      <div className="absolute inset-0 bg-overlay opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />

      <div className="flex justify-between items-start mb-6">
        <div>
          <span className="font-mono text-xs tracking-widest text-muted uppercase block mb-2">
            Selection No. 0{fragrance.rank}
          </span>
          <h3 className="font-sans text-sm tracking-widest uppercase text-muted mb-1">
            {fragrance.house}
          </h3>
          <h2 className="font-serif text-3xl text-ink leading-none">
            {fragrance.name}
          </h2>
        </div>

        <div className="text-right">
          <span className="font-mono text-xs tracking-widest text-muted uppercase block mb-1">
            Confidence
          </span>
          <span className="font-mono text-xl text-gold">
            {Math.round(fragrance.score * 100)}%
          </span>
        </div>
      </div>

      <div className="w-full h-px bg-border my-6" />

      <div className="mb-6">
        <span className="font-sans text-xs tracking-widest text-muted uppercase block mb-3">
          Key Accords
        </span>
        <div className="flex flex-wrap gap-2">
          {fragrance.notes.map((note) => (
            <span
              key={note}
              className="px-3 py-1 bg-bg border border-border text-ink text-xs font-sans tracking-wide"
            >
              {note}
            </span>
          ))}
        </div>
      </div>

      <div>
        <span className="font-sans text-xs tracking-widest text-muted uppercase block mb-2">
          Curator's Note — {fragrance.occasion}
        </span>
        <p className="font-serif text-lg leading-relaxed text-ink/80 italic">
          "{fragrance.reasoning}"
        </p>
      </div>
    </motion.div>
  );
}
