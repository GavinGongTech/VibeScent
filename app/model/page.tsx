"use client";

import { motion } from "framer-motion";
import PipelineVisual from "@/components/model/PipelineVisual";

export default function ModelPage() {
  const ease = [0.16, 1, 0.3, 1] as const;

  return (
    <div className="max-w-7xl mx-auto px-6 md:px-12 pt-32 pb-32 min-h-screen">
      {/* Page Hero */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease }}
        className="mb-24"
      >
        <span className="font-mono text-xs tracking-widest text-gold uppercase block mb-6">
          The Model
        </span>
        <h1 className="font-serif text-4xl md:text-6xl text-ink leading-tight max-w-3xl">
          Decoding aesthetics into olfactory signatures.
        </h1>
      </motion.div>

      {/* Pipeline Diagram */}
      <section className="mb-32">
        <h2 className="font-sans text-xs tracking-widest uppercase text-muted mb-8 border-b border-border pb-4">
          Architecture Pipeline
        </h2>
        <PipelineVisual />
      </section>

      {/* Architecture Notes */}
      <section className="mb-32">
        <h2 className="font-sans text-xs tracking-widest uppercase text-muted mb-12 border-b border-border pb-4">
          Methodology
        </h2>

        <div className="flex flex-col gap-16">
          {/* Note 1 */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6 md:gap-12">
            <div className="md:col-span-4">
              <h3 className="font-serif text-2xl text-ink">Vision Encoding</h3>
            </div>
            <div className="md:col-span-8">
              <p className="font-sans text-muted leading-relaxed text-lg font-light">
                The pipeline begins with a robust vision transformer (ViT) that
                analyzes the uploaded image. Rather than simply categorizing the
                garment, the model isolates textural features, color palettes,
                and structural silhouettes to establish a baseline visual
                aesthetic.
              </p>
            </div>
          </div>

          {/* Note 2 */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6 md:gap-12">
            <div className="md:col-span-4">
              <h3 className="font-serif text-2xl text-ink">
                Contextual Fusion
              </h3>
            </div>
            <div className="md:col-span-8">
              <p className="font-sans text-muted leading-relaxed text-lg font-light">
                Visual data is inherently subjective without context. The
                extracted image embeddings are fused with the optional
                categorical tags (Event, Time, Mood) provided by the user. This
                creates a multi-dimensional feature vector that represents the
                complete styling intent.
              </p>
            </div>
          </div>

          {/* Note 3 */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6 md:gap-12">
            <div className="md:col-span-4">
              <h3 className="font-serif text-2xl text-ink">
                Fragrance Mapping
              </h3>
            </div>
            <div className="md:col-span-8">
              <p className="font-sans text-muted leading-relaxed text-lg font-light">
                The fused vector is projected into a shared latent space
                alongside our curated dataset of luxury fragrances. Using cosine
                similarity, the model ranks the dataset, selecting the top three
                olfactory profiles that mathematically align with the user's
                visual aesthetic.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Tech Stack Table */}
      <section>
        <h2 className="font-sans text-xs tracking-widest uppercase text-muted mb-8 border-b border-border pb-4">
          Core Technologies
        </h2>

        <div className="w-full border border-border bg-surface/30 rounded-sm overflow-hidden">
          <table className="w-full text-left font-sans text-sm">
            <tbody>
              <tr className="border-b border-border">
                <th className="py-4 px-6 font-medium text-ink w-1/3">
                  Environment
                </th>
                <td className="py-4 px-6 text-muted font-light w-2/3">
                  Python 3.10, PyTorch
                </td>
              </tr>
              <tr className="border-b border-border">
                <th className="py-4 px-6 font-medium text-ink">Vision Model</th>
                <td className="py-4 px-6 text-muted font-light">
                  CLIP (Contrastive Language-Image Pretraining)
                </td>
              </tr>
              <tr className="border-b border-border">
                <th className="py-4 px-6 font-medium text-ink">
                  Data Processing
                </th>
                <td className="py-4 px-6 text-muted font-light">
                  Pandas, NumPy, Scikit-learn
                </td>
              </tr>
              <tr>
                <th className="py-4 px-6 font-medium text-ink">API Serving</th>
                <td className="py-4 px-6 text-muted font-light">
                  FastAPI (Future Implementation)
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
