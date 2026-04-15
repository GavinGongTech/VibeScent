"use client";

import { motion, Variants } from "framer-motion";
import Button from "../ui/Button";

export default function Hero() {
  // Editorial easing curve: cubic-bezier(0.16, 1, 0.3, 1)
  const ease: [number, number, number, number] = [0.16, 1, 0.3, 1];

  const containerVariants: Variants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.08, delayChildren: 0.2 },
    },
  };

  const itemVariants: Variants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 1, ease },
    },
  };

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden pt-20">
      {/* Subtle Grain Overlay (using CSS radial gradient for performance/mood) */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,var(--color-surface)_0%,var(--color-bg)_100%)] opacity-50 -z-10" />

      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="max-w-4xl mx-auto px-6 text-center z-10"
      >
        <motion.h1
          variants={itemVariants}
          className="font-serif text-5xl md:text-7xl lg:text-8xl leading-tight mb-6"
        >
          Dress for the occasion.
          <br />
          <span className="text-gold italic">Scent it perfectly.</span>
        </motion.h1>

        <motion.p
          variants={itemVariants}
          className="font-sans font-light text-muted text-lg md:text-xl max-w-2xl mx-auto mb-12"
        >
          An AI-driven olfactory curator that analyzes your attire and the
          moment to recommend the perfect luxury fragrance.
        </motion.p>

        <motion.div variants={itemVariants}>
          <Button href="/demo">Experience the Model</Button>
        </motion.div>
      </motion.div>
    </section>
  );
}
