"use client";

import { motion } from "framer-motion";

export default function AboutSection() {
  return (
    <section className="max-w-3xl mx-auto px-6 py-32 text-center">
      <motion.div
        initial={{ scaleX: 0 }}
        whileInView={{ scaleX: 1 }}
        viewport={{ once: true, margin: "-100px" }}
        transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
        className="h-px w-16 bg-gold mx-auto mb-12 origin-center"
      />
      <motion.p
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-100px" }}
        transition={{ duration: 1, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
        className="font-serif text-2xl md:text-3xl leading-relaxed text-ink/90"
      >
        VibeScent utilizes advanced vision encoders to map the textural and
        stylistic nuances of your outfit, fusing them with context to suggest
        fragrances that complete your aesthetic.
      </motion.p>
    </section>
  );
}
