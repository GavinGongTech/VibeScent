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
  // Graceful fallbacks in case the scraper misses something
  const price = fragrance.price || "Price Unavailable";
  const store = fragrance.store || "Retailer Unavailable";
  const hasLink = fragrance.purchaseUrl && fragrance.purchaseUrl !== "#";

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.8,
        ease: [0.16, 1, 0.3, 1],
        delay: index * 0.08,
      }}
      className="relative flex flex-col md:flex-row p-6 md:p-8 border border-border bg-surface group hover:border-gold/30 transition-colors duration-500 rounded-sm overflow-hidden gap-8"
    >
      {/* Subtle hover overlay */}
      <div className="absolute inset-0 bg-overlay opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />

      {/* Left Column: Product Thumbnail */}
      <div className="w-full md:w-1/3 shrink-0 flex items-center justify-center bg-white border border-border/50 rounded-sm overflow-hidden relative min-h-62.5">
        {fragrance.thumbnail ? (
          /* Using standard img tag because Next.js Image requires configuring external domains */
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={fragrance.thumbnail}
            alt={fragrance.name}
            className="object-contain w-full h-full max-h-62.5 p-4"
            referrerPolicy="no-referrer"
          />
        ) : (
          <span className="font-sans text-xs tracking-widest text-muted uppercase">
            No Image
          </span>
        )}
      </div>

      {/* Right Column: Details & Action */}
      <div className="w-full md:w-2/3 flex flex-col justify-between">
        <div>
          {/* Header */}
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
          </div>

          {/* Notes */}
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

          {/* Reasoning */}
          <div className="mb-8">
            <span className="font-sans text-xs tracking-widest text-muted uppercase block mb-2">
              Curator&apos;s Note — {fragrance.occasion}
            </span>
            <p className="font-serif text-lg leading-relaxed text-ink/80 italic">
              &quot;{fragrance.reasoning}&quot;
            </p>
          </div>
        </div>

        {/* Footer: Scraper Data & Button */}
        <div className="pt-6 border-t border-border flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <p className="font-mono text-xl text-ink mb-1">{price}</p>
            <p className="font-sans text-xs tracking-widest text-muted uppercase">
              {store}
            </p>
            {fragrance.inBudget === true && (
              <p className="font-mono text-xs tracking-widest text-gold uppercase mt-1">
                Within Budget
              </p>
            )}
          </div>

          {hasLink ? (
            <a
              href={fragrance.purchaseUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="px-8 py-3 bg-gold text-bg font-sans text-xs tracking-widest uppercase hover:bg-gold-dim transition-colors cursor-pointer rounded-sm text-center w-full sm:w-auto"
            >
              Acquire Scent
            </a>
          ) : (
            <button
              disabled
              className="px-8 py-3 bg-muted/20 text-muted font-sans text-xs tracking-widest uppercase cursor-not-allowed rounded-sm text-center w-full sm:w-auto"
            >
              Unavailable
            </button>
          )}
        </div>
      </div>
    </motion.div>
  );
}
