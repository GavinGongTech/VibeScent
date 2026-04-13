"use client";

interface TagProps {
  label: string;
  selected: boolean;
  onClick: () => void;
}

export default function Tag({ label, selected, onClick }: TagProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`px-4 py-2 text-xs uppercase tracking-widest font-sans font-medium transition-all duration-300 border ${
        selected
          ? "bg-gold text-bg border-gold"
          : "bg-transparent text-muted border-border hover:border-gold/50 hover:text-ink"
      }`}
    >
      {label}
    </button>
  );
}
