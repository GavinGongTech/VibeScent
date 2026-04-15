"use client";

import { useState, useRef, DragEvent, ChangeEvent } from "react";
import { motion } from "framer-motion";

interface OutfitUploaderProps {
  onImageReady: (base64: string, mimeType: string) => void;
}

export default function OutfitUploader({ onImageReady }: OutfitUploaderProps) {
  const [dragActive, setDragActive] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const processFile = (file: File) => {
    setError(null);

    // Validate type
    const validTypes = ["image/jpeg", "image/png", "image/webp"];
    if (!validTypes.includes(file.type)) {
      setError("Please upload a JPG, PNG, or WEBP image.");
      return;
    }

    // Validate size (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
      setError("File size exceeds 10MB limit.");
      return;
    }

    const reader = new FileReader();
    reader.onloadend = () => {
      const base64String = reader.result as string;
      setPreviewUrl(base64String);
      onImageReady(base64String, file.type);
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const triggerSelect = () => {
    inputRef.current?.click();
  };

  const clearImage = (e: React.MouseEvent) => {
    e.stopPropagation();
    setPreviewUrl(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="w-full">
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        accept="image/jpeg, image/png, image/webp"
        onChange={handleChange}
      />

      {!previewUrl ? (
        <div
          className={`relative flex flex-col items-center justify-center w-full h-100 rounded-sm border border-dashed transition-all duration-300 cursor-pointer ${
            dragActive
              ? "border-gold bg-overlay"
              : "border-border bg-surface hover:border-gold/50 hover:bg-surface/80"
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={triggerSelect}
        >
          <div className="text-center p-6 flex flex-col items-center">
            <div className="w-12 h-12 mb-6 rounded-full bg-bg flex items-center justify-center border border-border">
              <svg
                className="w-5 h-5 text-gold"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
                />
              </svg>
            </div>
            <p className="font-sans font-medium text-ink mb-2">
              Click or drag outfit here
            </p>
            <p className="font-sans text-sm tracking-wide text-muted uppercase">
              JPG, PNG, WEBP up to 10MB
            </p>

            {error && (
              <motion.p
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-6 font-mono text-xs tracking-wide text-bg bg-gold px-3 py-1 rounded-sm"
              >
                {error}
              </motion.p>
            )}
          </div>
        </div>
      ) : (
        <motion.div
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          className="relative w-full aspect-3/4 group rounded-sm overflow-hidden border border-gold"
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={previewUrl}
            alt="Outfit preview"
            className="w-full h-full object-cover"
          />
          <div
            className="absolute inset-0 bg-bg/80 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center backdrop-blur-sm cursor-pointer"
            onClick={triggerSelect}
          >
            <div className="flex flex-col gap-4 items-center">
              <p className="font-sans text-sm tracking-widest uppercase text-ink">
                Change Image
              </p>
              <button
                onClick={clearImage}
                className="font-sans text-xs tracking-widest uppercase text-muted hover:text-gold transition-colors"
              >
                Remove
              </button>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}
