"use client";

import { motion } from "framer-motion";
import PipelineVisual from "@/components/model/PipelineVisual";
import DataVisual from "@/components/model/DataVisual";

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
              <h3 className="font-serif text-2xl text-ink">Embedding</h3>
            </div>
            <div className="md:col-span-8">
              <p className="font-sans text-muted leading-relaxed text-lg font-light">
                When the user uploads their outfit and context, the data is
                immediately split into distinct processing tracks. The image is
                passed through a vision transformer (CLIP) to convert visual
                data into embeddings. The text is simultaneously passed through
                a sentence transformer (Qwen3-Embedding) to create embeddings
                that represent context and the user&apos;s intent.
              </p>
            </div>
          </div>

          {/* Note 2 */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6 md:gap-12">
            <div className="md:col-span-4">
              <h3 className="font-serif text-2xl text-ink">
                Translation to Vibe Space
              </h3>
            </div>
            <div className="md:col-span-8">
              <p className="font-sans text-muted leading-relaxed text-lg font-light">
                The image embeddings are passed through (CLIP) to represent
                vibes. The text embeddings are passed through an LLM to
                represent vibes. A multimodal LLM (Qwen 3.5) processes
                embeddings from both the text and image to understand the
                relationship between the outfit and the context.
              </p>
            </div>
          </div>

          {/* Note 3 */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6 md:gap-12">
            <div className="md:col-span-4">
              <h3 className="font-serif text-2xl text-ink">Average Vibe</h3>
            </div>
            <div className="md:col-span-8">
              <p className="font-sans text-muted leading-relaxed text-lg font-light">
                With multiple vibes, the model computes a dynamically weighted
                sum. These weights are learned to prioritize vibes that
                contribute the most meaning to the model. The result is an
                average vibe across inputs.
              </p>
            </div>
          </div>

          {/* Note 4 */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6 md:gap-12">
            <div className="md:col-span-4">
              <h3 className="font-serif text-2xl text-ink">
                Fragrance Comparison
              </h3>
            </div>
            <div className="md:col-span-8">
              <p className="font-sans text-muted leading-relaxed text-lg font-light">
                The model executes a similarity search using cosine similarity
                to identify fragrances with similar vibes. The top-p fragrances
                closest to the user&apos;s desired aesthetic are recommended to the
                user.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Data Pipeline Diagram */}
      <section className="mb-32">
        <h2 className="font-sans text-xs tracking-widest uppercase text-muted mb-8 border-b border-border pb-4">
          Data Preparations
        </h2>
        <DataVisual />
      </section>

      {/* Data Notes */}
      <section className="mb-32">
        <h2 className="font-sans text-xs tracking-widest uppercase text-muted mb-12 border-b border-border pb-4">
          Steps
        </h2>

        <div className="flex flex-col gap-16">
          {/* Note 1 */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6 md:gap-12">
            <div className="md:col-span-4">
              <h3 className="font-serif text-2xl text-ink">Data Preparation</h3>
            </div>
            <div className="md:col-span-8">
              <p className="font-sans text-muted leading-relaxed text-lg font-light">
                Through normalization, deduplication, and cross-referencing, the
                system merges and consolidates 5 datasets into a single, unified
                dataset containing 36,000 unique fragrances with metadata and
                scent data.
              </p>
            </div>
          </div>

          {/* Note 2 */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6 md:gap-12">
            <div className="md:col-span-4">
              <h3 className="font-serif text-2xl text-ink">
                Feature Extraction
              </h3>
            </div>
            <div className="md:col-span-8">
              <p className="font-sans text-muted leading-relaxed text-lg font-light">
                The system isolates the top, middle, and bass notes alongside
                the main accords that describe each fragrance.
              </p>
            </div>
          </div>

          {/* Note 3 */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6 md:gap-12">
            <div className="md:col-span-4">
              <h3 className="font-serif text-2xl text-ink">Embedding</h3>
            </div>
            <div className="md:col-span-8">
              <p className="font-sans text-muted leading-relaxed text-lg font-light">
                To allow the machine to understand scent, the extracted notes
                and accords are passed through a sentence transformer
                (Qwen3-Embedding) to convert textual descriptions into vectors
                that, map the spatial and semantic relationships between
                different fragrance profiles.
              </p>
            </div>
          </div>

          {/* Note 4 */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6 md:gap-12">
            <div className="md:col-span-4">
              <h3 className="font-serif text-2xl text-ink">Fragrance Vibes</h3>
            </div>
            <div className="md:col-span-8">
              <p className="font-sans text-muted leading-relaxed text-lg font-light">
                Embedded vectors are processed through an LLM to project them
                into a Vibe Space. Each vibe includes the following features:
                formality, seasonality, gender, and time of day.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Tech Stack Table
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
      </section> */}
    </div>
  );
}
