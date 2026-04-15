"use client";

import { motion } from "framer-motion";

interface SubmitButtonProps {
  onClick: () => void;
  disabled: boolean;
  loading: boolean;
}

export default function SubmitButton({
  onClick,
  disabled,
  loading,
}: SubmitButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled || loading}
      className="relative w-full py-4 mt-6 flex items-center justify-center bg-gold text-bg font-sans font-medium tracking-widest uppercase transition-all duration-300 hover:bg-gold-dim focus:outline-none focus:ring-2 focus:ring-gold/50 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer overflow-hidden rounded-sm"
    >
      {loading ? (
        <div className="flex items-center gap-3">
          {/* Custom minimal spinner */}
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
            className="w-4 h-4 border-2 border-bg border-t-transparent rounded-full"
          />
          <span>Analysing your look...</span>
        </div>
      ) : (
        <span>Curate My Fragrance</span>
      )}
    </button>
  );
}
