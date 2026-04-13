"use client";

import { motion, Variants } from "framer-motion";

export default function PipelineVisual() {
  const steps = [
    { title: "Outfit Image", sub: "User Upload (Base64)" },
    { title: "Vision Encoder", sub: "Feature Extraction (CLIP/ViT)" },
    { title: "Feature Fusion", sub: "Contextual Weighting" },
    { title: "Fragrance Ranker", sub: "Similarity Scoring" },
    { title: "Top 3 Results", sub: "Ranked JSON Response" },
  ];

  const containerVariants: Variants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.15 },
    },
  };

  const itemVariants: Variants = {
    hidden: { opacity: 0, x: -20 },
    visible: {
      opacity: 1,
      x: 0,
      transition: { duration: 0.8, ease: [0.16, 1, 0.3, 1] },
    },
  };

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: "-100px" }}
      className="w-full overflow-x-auto pb-8 hide-scrollbar"
    >
      <div className="flex items-center min-w-max py-4">
        {steps.map((step, index) => (
          <div key={step.title} className="flex items-center">
            {/* The Node */}
            <motion.div
              variants={itemVariants}
              className="flex flex-col justify-center h-24 px-6 border border-border bg-surface rounded-sm min-w-50"
            >
              <span className="font-mono text-xs text-gold mb-2 tracking-widest uppercase">
                Step 0{index + 1}
              </span>
              <h3 className="font-sans text-sm text-ink font-medium tracking-wide">
                {step.title}
              </h3>
              <p className="font-sans text-xs text-muted mt-1">{step.sub}</p>
            </motion.div>

            {/* The Arrow (Don't render after the last item) */}
            {index < steps.length - 1 && (
              <motion.div variants={itemVariants} className="px-4 text-border">
                <svg
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <line x1="5" y1="12" x2="19" y2="12"></line>
                  <polyline points="12 5 19 12 12 19"></polyline>
                </svg>
              </motion.div>
            )}
          </div>
        ))}
      </div>
    </motion.div>
  );
}
