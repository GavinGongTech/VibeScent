"use client";

import { ContextInput } from "@/lib/types";

interface ContextFormProps {
  value: ContextInput;
  onChange: (value: ContextInput) => void;
  budget: number;
  onBudgetChange: (budget: number) => void;
}

const EVENT_TYPES = ["Gala", "Date Night", "Casual", "Business", "Wedding", "Festival"];
const TIMES_OF_DAY = ["Morning", "Afternoon", "Evening", "Night"];
const MOODS = ["Bold", "Subtle", "Fresh", "Warm", "Mysterious"];

function PillGroup({
  label,
  options,
  selected,
  onSelect,
}: {
  label: string;
  options: string[];
  selected: string | undefined;
  onSelect: (val: string | undefined) => void;
}) {
  return (
    <div className="flex flex-col gap-3">
      <span className="font-sans text-xs font-medium tracking-widest uppercase text-muted">
        {label}
      </span>
      <div className="flex flex-wrap gap-2">
        {options.map((opt) => {
          const active = selected === opt;
          return (
            <button
              key={opt}
              type="button"
              onClick={() => onSelect(active ? undefined : opt)}
              className={[
                "px-3 py-1.5 rounded-sm border text-xs font-sans font-medium tracking-wide transition-colors duration-200",
                active
                  ? "border-gold text-gold bg-gold/10"
                  : "border-border text-muted hover:border-gold/50 hover:text-ink",
              ].join(" ")}
            >
              {opt}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export default function ContextForm({ value, onChange, budget, onBudgetChange }: ContextFormProps) {
  return (
    <div className="w-full flex flex-col gap-8">
      <PillGroup
        label="Event Type"
        options={EVENT_TYPES}
        selected={value.eventType}
        onSelect={(v) => onChange({ ...value, eventType: v })}
      />
      <PillGroup
        label="Time of Day"
        options={TIMES_OF_DAY}
        selected={value.timeOfDay}
        onSelect={(v) => onChange({ ...value, timeOfDay: v })}
      />
      <PillGroup
        label="Mood"
        options={MOODS}
        selected={value.mood}
        onSelect={(v) => onChange({ ...value, mood: v })}
      />
      <div className="flex flex-col gap-3">
        <div className="flex justify-between items-center">
          <span className="font-sans text-xs font-medium tracking-widest uppercase text-muted">
            Max Budget
          </span>
          <span className="font-mono text-sm text-gold">${budget}</span>
        </div>
        <input
          type="range"
          min="50"
          max="1000"
          step="10"
          value={budget}
          onChange={(e) => onBudgetChange(Number(e.target.value))}
          className="w-full h-1 bg-border rounded-lg appearance-none cursor-pointer accent-gold"
        />
      </div>
    </div>
  );
}
