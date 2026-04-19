"use client";

interface ContextFormProps {
  value: string;
  onChange: (value: string) => void;
}

export default function ContextForm({ value, onChange }: ContextFormProps) {
  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    onChange(e.target.value);
  };

  return (
    <div className="w-full flex flex-col gap-10">
      <div className="flex flex-col gap-4 mt-2">
        <label
          htmlFor="event-description"
          className="font-serif text-xl text-ink"
        >
          Event Description
        </label>
        <textarea
          id="event-description"
          value={value}
          onChange={handleTextChange}
          placeholder="Describe the venue, specific scent notes you love, or the exact vibe you are going for..."
          className="w-full bg-surface border border-border rounded-sm p-4 text-ink font-sans text-sm focus:outline-none focus:border-gold transition-colors duration-300 placeholder:text-muted resize-none shadow-sm"
          rows={4}
        />
      </div>
    </div>
  );
}
