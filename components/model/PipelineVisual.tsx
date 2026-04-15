"use client";

import { motion, Variants } from "framer-motion";

// Define the shape of a single node and a stage (which can hold multiple nodes)
interface WorkflowNode {
  title: string;
  sub: string;
}

type Stage = WorkflowNode[];

export default function PipelineVisual() {
  // ── THE WORKFLOW DATA ───────────────────────────────────────────
  // Each array inside `stages` represents a column in the visualizer.
  // Add multiple objects to an array to create a branching/parallel path.
  const stages: Stage[] = [
    [{ title: "Image + Text Inputs", sub: "User Upload" }],
    [
      { title: "Vision Encoder", sub: "Extract Image Embeddings (CLIP)" },
      {
        title: "Context Parser",
        sub: "Extract Text Embeddings (Qwen3-Embedding)",
      },
    ],
    [
      { title: "Image Vibes", sub: "Image Embeddings → Vibe Space (CLIP)" },
      { title: "Context Vibes", sub: "Text Embeddings → Vibe Space (LLM)" },
      {
        title: "Multimodal Vibes",
        sub: "Combined Embeddings → Vibe Space (Qwen-3.5)",
      },
    ],
    [{ title: "Average Vibes", sub: "Weighted Sum of Vibes" }],
    [
      {
        title: "Fragrance Comparison",
        sub: "Match Existing Fragrances",
      },
    ],
    [{ title: "Final Output", sub: "Top Recommended Fragrances" }],
  ];
  // ────────────────────────────────────────────────────────────────

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
      <div className="flex items-center min-w-max py-4 px-2">
        {stages.map((stage, stageIndex) => (
          <div key={`stage-${stageIndex}`} className="flex items-center">
            {/* The Column (Handles 1 or multiple nodes) */}
            <div className="flex flex-col gap-4">
              {stage.map((node, nodeIndex) => (
                <motion.div
                  key={node.title}
                  variants={itemVariants}
                  className="flex flex-col justify-center h-24 px-6 border border-border bg-surface rounded-sm min-w-55"
                >
                  <span className="font-mono text-xs text-gold mb-2 tracking-widest uppercase">
                    {`Step 0${stageIndex + 1}`}
                  </span>
                  <h3 className="font-sans text-sm text-ink font-medium tracking-wide">
                    {node.title}
                  </h3>
                  <p className="font-sans text-xs text-muted mt-1">
                    {node.sub}
                  </p>
                </motion.div>
              ))}
            </div>

            {/* The Arrow (Renders between columns, not after the last one) */}
            {stageIndex < stages.length - 1 && (
              <motion.div
                variants={itemVariants}
                className="px-6 text-border flex items-center justify-center"
              >
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
                  <line x1="0" y1="12" x2="19" y2="12"></line>
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
