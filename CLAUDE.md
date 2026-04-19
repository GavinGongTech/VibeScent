# CLAUDE.md — ScentAI Frontend

> This file is the single source of truth for any AI assistant or developer working on this project.
> Read it fully before writing any code, creating any component, or modifying any existing file.

---

## Project Overview

**ScentAI** is a portfolio/demo web application that showcases an ML model capable of recommending fragrances based on an uploaded outfit image and optional event context. The frontend is the primary artifact — it must feel high-fashion, editorial, and cinematic while making the ML model the hero of the experience.

The model is not yet integrated. All model-facing code lives behind a single API stub (`/api/recommend`) so the model can be dropped in later without touching the UI.

---

## Aesthetic Direction

**Vibe:** High-fashion editorial. Think Byredo, Aesop, or a luxury fragrance editorial spread — dark, moody, and considered. Every spacing, font, and color decision should feel intentional and luxurious.

### Color Palette

| Token              | Value                    | Usage                                         |
| ------------------ | ------------------------ | --------------------------------------------- |
| `--color-bg`       | `#0a0a0a`                | Page background                               |
| `--color-surface`  | `#111111`                | Cards, panels, elevated surfaces              |
| `--color-border`   | `#2a2a2a`                | Subtle borders, dividers                      |
| `--color-gold`     | `#c9a96e`                | Primary accent — CTAs, highlights, scores     |
| `--color-gold-dim` | `#8b6914`                | Secondary gold — hover states, subtitles      |
| `--color-text`     | `#f5f0e8`                | Primary text (warm off-white, not pure white) |
| `--color-muted`    | `#888070`                | Secondary text, labels, placeholders          |
| `--color-overlay`  | `rgba(201,169,110,0.06)` | Hover overlays on cards                       |

### Typography

| Role           | Font               | Weight  | Notes                                      |
| -------------- | ------------------ | ------- | ------------------------------------------ |
| Display / H1   | Cormorant Garamond | 300–400 | Large, elegant serif for hero text         |
| Headings H2–H3 | Cormorant Garamond | 500     | Slightly more weighted for section headers |
| Body / UI      | DM Sans            | 300–400 | Clean, modern counterpart to the serif     |
| Labels / Caps  | DM Sans            | 500     | Letter-spaced uppercase for tags/labels    |
| Monospace      | DM Mono            | 400     | Confidence scores, technical readouts      |

Load all fonts from Google Fonts. No fallback to Inter, Arial, or system fonts.

```html
<link
  href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400&display=swap"
  rel="stylesheet"
/>
```

### Motion Principles

- Use **Framer Motion** for all page-level and component-level animations.
- Default easing: `[0.16, 1, 0.3, 1]` (expo out) — smooth, editorial feel.
- Stagger children with `0.08s` delay increments.
- Fade + translate-up (`y: 20 → 0`) for all element entrances.
- Results panel uses a reveal animation: cards stagger in from bottom with opacity 0 → 1.
- No bounce, spring, or playful physics — everything should feel deliberate and refined.
- Respect `prefers-reduced-motion`.

---

## Tech Stack

| Layer       | Technology           | Version | Notes                                                    |
| ----------- | -------------------- | ------- | -------------------------------------------------------- |
| Framework   | Next.js (App Router) | 14+     | Use `app/` directory, server components where applicable |
| Styling     | Tailwind CSS         | 3+      | Extend with custom design tokens in `tailwind.config.ts` |
| Animation   | Framer Motion        | 11+     | All transitions and reveals                              |
| Language    | TypeScript           | 5+      | Strict mode enabled                                      |
| Deployment  | Vercel               | —       | Auto-deploy from `main` branch                           |
| Package Mgr | npm                  | —       | Use `npm install`, not yarn or pnpm                      |

---

## Project Structure

```
scentai/
├── app/
│   ├── layout.tsx               # Root layout: fonts, metadata, Navbar, dark background
│   ├── page.tsx                 # Landing page (/)
│   ├── demo/
│   │   └── page.tsx             # Demo page (/demo)
│   ├── model/
│   │   └── page.tsx             # How it works page (/model)
│   └── api/
│       └── recommend/
│           └── route.ts         # POST /api/recommend — model stub (see below)
│
├── components/
│   ├── ui/                      # Primitive design system components
│   │   ├── Button.tsx
│   │   ├── Tag.tsx              # Pill/tag selector for context inputs
│   │   └── GoldDivider.tsx      # Thin gold horizontal rule
│   ├── layout/
│   │   ├── Navbar.tsx           # Minimal top nav: logo left, links right
│   │   └── Footer.tsx
│   ├── landing/
│   │   ├── Hero.tsx             # Full-bleed cinematic hero section
│   │   └── AboutSection.tsx     # Brief model description below hero
│   ├── demo/
│   │   ├── OutfitUploader.tsx   # Drag-and-drop image upload with preview
│   │   ├── ContextForm.tsx      # Optional event/time/mood inputs
│   │   ├── SubmitButton.tsx     # Animated CTA with loading state
│   │   └── ResultsPanel.tsx     # Top 3 fragrance recommendation cards
│   └── model/
│       └── PipelineVisual.tsx   # ML pipeline diagram / explainer
│
├── lib/
│   ├── types.ts                 # All shared TypeScript types (see below)
│   └── recommend.ts             # Client-side fetch wrapper for /api/recommend
│
├── styles/
│   └── globals.css              # CSS variables, base resets, font declarations
│
├── public/
│   └── ...                      # Static assets
│
├── tailwind.config.ts           # Extend with design tokens
├── CLAUDE.md                    # This file
└── tsconfig.json
```

---

## Page Descriptions

### `/` — Landing Page

The entry point. Cinematic, minimal, and editorial. Goal: communicate what ScentAI is in under five seconds and compel the visitor to try the demo.

**Sections:**

1. **Navbar** — Logo (`SCENTAI`) left, nav links (`Demo`, `Model`) right. Transparent background over hero, blurs on scroll.
2. **Hero** — Full viewport height. Large Cormorant Garamond headline split across two lines (e.g. `"Dress for the occasion. / Scent it perfectly."`). Subline in DM Sans 300. Single gold CTA button → `/demo`. Subtle dark-grain texture overlay on background.
3. **About strip** — 2–3 sentences explaining the model. Gold accent line above it.
4. **Footer** — Minimal. Logo, year, optional GitHub link.

**Do not** add feature grids, testimonials, or marketing copy. Keep it sparse.

---

### `/demo` — Demo Page

The primary interactive experience. Two-column layout on desktop (inputs left, results right). Single column on mobile.

**Left column — Inputs:**

1. `OutfitUploader` — Large dashed-border drop zone. On upload, shows image preview with a subtle gold border. Accepts JPG/PNG/WEBP up to 10MB.
2. `ContextForm` — Three optional input groups:
   - **Event type:** pill selectors — `Gala`, `Date Night`, `Casual`, `Business`, `Wedding`, `Festival`
   - **Time of day:** pill selectors — `Morning`, `Afternoon`, `Evening`, `Night`
   - **Mood:** pill selectors — `Bold`, `Subtle`, `Fresh`, `Warm`, `Mysterious`
3. `SubmitButton` — Full-width gold button. Shows spinner + "Analysing your look…" while awaiting response.

**Right column — Results:**

- Initially shows a placeholder state: faint dashed border, copy like `"Your recommendation will appear here."` in muted text.
- On result: `ResultsPanel` animates in with three `FragranceCard` components (see types below), staggered 80ms apart.

---

### `/model` — How It Works

An editorial explainer page for recruiters and professors. Not a wall of text.

**Sections:**

1. **Page hero** — Small-caps label `"The Model"`, large Cormorant heading.
2. **Pipeline visual** — `PipelineVisual` component: a horizontal flow diagram showing `Outfit Image → Vision Encoder → Feature Fusion → Fragrance Ranker → Top 3 Results`. Can be a styled SVG or a series of animated cards.
3. **Architecture notes** — Two or three short prose paragraphs describing the approach. Use a two-column grid on desktop (label left, body right).
4. **Tech stack table** — Model-side technologies (e.g. PyTorch, CLIP, etc.) listed cleanly.

---

## Component Details

### `OutfitUploader`

```tsx
interface OutfitUploaderProps {
  onImageReady: (base64: string, mimeType: string) => void;
}
```

- Uses the HTML5 drag-and-drop API + a hidden `<input type="file">`.
- Converts the selected file to base64 via `FileReader` and calls `onImageReady`.
- Shows a gold-bordered preview thumbnail after upload.
- Shows an error state if file is too large (>10MB) or wrong type.

---

### `ContextForm`

```tsx
interface ContextFormProps {
  value: ContextInput;
  onChange: (value: ContextInput) => void;
}
```

All fields are optional. Submitting with no context is valid — the model should handle it gracefully.

---

### `ResultsPanel`

```tsx
interface ResultsPanelProps {
  results: FragranceRecommendation[] | null;
  loading: boolean;
}
```

When `loading` is true, shows three skeleton cards. When `results` is populated, animates in the real cards. When `results` is null and `loading` is false, shows the placeholder state.

---

## TypeScript Types

All shared types live in `lib/types.ts`.

```typescript
// Input sent to the model
export interface RecommendRequest {
  image: string; // base64-encoded image data
  mimeType: string; // "image/jpeg" | "image/png" | "image/webp"
  context: ContextInput;
}

export interface ContextInput {
  eventType?: string; // e.g. "Gala", "Date Night"
  timeOfDay?: string; // e.g. "Evening"
  mood?: string; // e.g. "Mysterious"
}

// Output returned by the model
export interface RecommendResponse {
  recommendations: FragranceRecommendation[];
}

export interface FragranceRecommendation {
  rank: number; // 1, 2, or 3
  name: string; // Fragrance name e.g. "Bleu de Chanel"
  house: string; // Brand e.g. "Chanel"
  score: number; // Confidence 0.0–1.0
  notes: string[]; // Top scent notes e.g. ["cedarwood", "amber", "musk"]
  reasoning: string; // 1–2 sentence explanation from the model
  occasion: string; // Short occasion label e.g. "Black tie evening"
}
```

---

## Model Integration — `/api/recommend`

**This is the only file that changes when the real model is plugged in.**

Current stub (`app/api/recommend/route.ts`):

```typescript
import { NextRequest, NextResponse } from "next/server";
import type { RecommendRequest, RecommendResponse } from "@/lib/types";

export async function POST(req: NextRequest) {
  const body: RecommendRequest = await req.json();

  // ── STUB ──────────────────────────────────────────────────────────────
  // Replace this block with your real model call.
  // The body contains: body.image (base64), body.mimeType, body.context
  // Return a RecommendResponse-shaped object.
  // ──────────────────────────────────────────────────────────────────────

  const mockResponse: RecommendResponse = {
    recommendations: [
      {
        rank: 1,
        name: "Baccarat Rouge 540",
        house: "Maison Francis Kurkdjian",
        score: 0.91,
        notes: ["jasmine", "saffron", "amberwood", "fir resin"],
        reasoning:
          "The structured elegance of the outfit calls for a warm, luminous signature. Baccarat Rouge 540's crystalline amber and saffron accord mirrors the refined opulence of your look.",
        occasion: "Formal evening event",
      },
      {
        rank: 2,
        name: "Black Orchid",
        house: "Tom Ford",
        score: 0.84,
        notes: ["black truffle", "ylang ylang", "dark chocolate", "patchouli"],
        reasoning:
          "The deep tones in your palette align with this fragrance's dark floral complexity — a bold, commanding presence for a high-stakes occasion.",
        occasion: "Cocktail or gala",
      },
      {
        rank: 3,
        name: "Portrait of a Lady",
        house: "Frédéric Malle",
        score: 0.76,
        notes: ["turkish rose", "raspberry", "patchouli", "incense"],
        reasoning:
          "For a softer interpretation of the look, this rose-centered oriental brings warmth and depth without overpowering the visual statement you're making.",
        occasion: "Evening dinner",
      },
    ],
  };

  return NextResponse.json(mockResponse);
}
```

**When integrating the real model:**

- Import your model inference function or call your model's hosted API endpoint.
- Pass `body.image`, `body.mimeType`, and `body.context` to it.
- Map its output to the `RecommendResponse` shape.
- Return `NextResponse.json(result)`.
- Do not touch any component, page, or type file.

---

## Tailwind Config Extensions

Add the following to `tailwind.config.ts` to wire design tokens into utility classes:

```typescript
theme: {
  extend: {
    colors: {
      bg: '#0a0a0a',
      surface: '#111111',
      border: '#2a2a2a',
      gold: '#c9a96e',
      'gold-dim': '#8b6914',
      ink: '#f5f0e8',
      muted: '#888070',
    },
    fontFamily: {
      serif: ['"Cormorant Garamond"', 'Georgia', 'serif'],
      sans: ['"DM Sans"', 'sans-serif'],
      mono: ['"DM Mono"', 'monospace'],
    },
    letterSpacing: {
      widest: '0.2em',
    },
  },
}
```

---

## Build Order

Build features in this sequence. Each skill is self-contained and builds on the previous.

| #   | Skill / Feature    | Output                                                           |
| --- | ------------------ | ---------------------------------------------------------------- |
| 1   | Design system      | `globals.css`, `tailwind.config.ts`, font setup                  |
| 2   | Landing page       | `app/page.tsx`, `Hero.tsx`, `Navbar.tsx`, `Footer.tsx`           |
| 3   | Image uploader     | `OutfitUploader.tsx`                                             |
| 4   | Context form       | `ContextForm.tsx`, `Tag.tsx`                                     |
| 5   | Results panel      | `ResultsPanel.tsx`, `FragranceCard.tsx`                          |
| 6   | Model API stub     | `app/api/recommend/route.ts`, `lib/types.ts`, `lib/recommend.ts` |
| 7   | Demo page assembly | `app/demo/page.tsx` — wires skills 3–6 together                  |
| 8   | How it works page  | `app/model/page.tsx`, `PipelineVisual.tsx`                       |

---

## Code Conventions

- **No `any` types.** Use the types in `lib/types.ts` throughout.
- **Server vs client components.** Default to server components. Add `"use client"` only when the component uses state, effects, or browser APIs.
- **Tailwind over inline styles.** Use utility classes. Reserve `style={{}}` only for dynamic values that can't be expressed as classes (e.g. computed widths, animation delays).
- **Component files are single-responsibility.** One component per file. No mega-files.
- **All images handled as base64 strings** within the app. Do not write image files to disk.
- **Responsive breakpoints:** Mobile-first. Key breakpoint is `md` (768px) for the two-column demo layout.
- **Accessibility:** All interactive elements must be keyboard-accessible. Images need `alt` text. Color is never the only indicator of state.

---

## What NOT to Do

- Do not add a database or authentication — this is a stateless demo.
- Do not store uploaded images anywhere — process in memory only.
- Do not use purple gradients, glassmorphism, or neon glows.
- Do not use Inter, Roboto, or system fonts.
- Do not add loading skeletons that feel "techy" — keep them minimal and on-brand (dark surface, subtle shimmer in gold).
- Do not add marketing sections, testimonials, or feature grids to the landing page.
- Do not build a multi-step wizard for the demo — all inputs are visible at once.
- Do not change `lib/types.ts` without updating every consuming component.

---

## Running the Project

```bash
npm install
npm run dev       # http://localhost:3000
npm run build     # production build check before deploying
```

Deploy by pushing to `main` on GitHub — Vercel auto-deploys.

---

_Last updated: project inception. Update this file whenever the model integration contract changes or a new page is added._

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)
