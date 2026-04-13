"use client";

import { ContextInput } from "@/lib/types";
import Tag from "../ui/Tag";

interface ContextFormProps {
  value: ContextInput;
  onChange: (value: ContextInput) => void;
}

export default function ContextForm({ value, onChange }: ContextFormProps) {
  const categories = {
    eventType: [
      "Gala",
      "Date Night",
      "Casual",
      "Business",
      "Wedding",
      "Festival",
    ],
    timeOfDay: ["Morning", "Afternoon", "Evening", "Night"],
    mood: ["Bold", "Subtle", "Fresh", "Warm", "Mysterious"],
  };

  const handleToggle = (category: keyof ContextInput, tag: string) => {
    const newValue = value[category] === tag ? undefined : tag;
    onChange({ ...value, [category]: newValue });
  };

  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    onChange({ ...value, customNotes: e.target.value });
  };

  return (
    <div className="w-full flex flex-col gap-10">
      {/* Event Type */}
      <div className="flex flex-col gap-4">
        <label className="font-serif text-xl text-ink">Event Context</label>
        <div className="flex flex-wrap gap-3">
          {categories.eventType.map((tag) => (
            <Tag
              key={tag}
              label={tag}
              selected={value.eventType === tag}
              onClick={() => handleToggle("eventType", tag)}
            />
          ))}
        </div>
      </div>

      {/* Time of Day */}
      <div className="flex flex-col gap-4">
        <label className="font-serif text-xl text-ink">Time of Day</label>
        <div className="flex flex-wrap gap-3">
          {categories.timeOfDay.map((tag) => (
            <Tag
              key={tag}
              label={tag}
              selected={value.timeOfDay === tag}
              onClick={() => handleToggle("timeOfDay", tag)}
            />
          ))}
        </div>
      </div>

      {/* Mood */}
      <div className="flex flex-col gap-4">
        <label className="font-serif text-xl text-ink">Desired Aura</label>
        <div className="flex flex-wrap gap-3">
          {categories.mood.map((tag) => (
            <Tag
              key={tag}
              label={tag}
              selected={value.mood === tag}
              onClick={() => handleToggle("mood", tag)}
            />
          ))}
        </div>
      </div>

      {/* Custom Notes */}
      <div className="flex flex-col gap-4 mt-2">
        <label htmlFor="custom-notes" className="font-serif text-xl text-ink">
          Additional Details
        </label>
        <textarea
          id="custom-notes"
          value={value.customNotes || ""}
          onChange={handleTextChange}
          placeholder="Describe the venue, specific scent notes you love, or the exact vibe you are going for..."
          className="w-full bg-transparent border-b border-border text-ink font-sans text-sm focus:outline-none focus:border-gold transition-colors duration-300 placeholder:text-muted resize-none py-3"
          rows={2}
        />
      </div>
    </div>
  );
}
