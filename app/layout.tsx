import React from "react";
import type { Metadata } from "next";
import localFont from "next/font/local";
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import "../styles/globals.css";

// Configure requested fonts and weights
const cormorant = localFont({
  src: [
    {
      path: "../assets/fonts/cormorant-garamond-300.ttf",
      weight: "300",
      style: "normal",
    },
    {
      path: "../assets/fonts/cormorant-garamond-400.ttf",
      weight: "400",
      style: "normal",
    },
    {
      path: "../assets/fonts/cormorant-garamond-500.ttf",
      weight: "500",
      style: "normal",
    },
  ],
  variable: "--font-cormorant",
  display: "swap",
});

const dmSans = localFont({
  src: [
    {
      path: "../assets/fonts/dm-sans-300.ttf",
      weight: "300",
      style: "normal",
    },
    {
      path: "../assets/fonts/dm-sans-400.ttf",
      weight: "400",
      style: "normal",
    },
    {
      path: "../assets/fonts/dm-sans-500.ttf",
      weight: "500",
      style: "normal",
    },
  ],
  variable: "--font-dm-sans",
  display: "swap",
});

const dmMono = localFont({
  src: [
    {
      path: "../assets/fonts/dm-mono-400.ttf",
      weight: "400",
      style: "normal",
    },
  ],
  variable: "--font-dm-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "VibeScent | High-Fashion Fragrance Recommendations",
  description: "Dress for the occasion. Scent it perfectly.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${cormorant.variable} ${dmSans.variable} ${dmMono.variable}`}
      suppressHydrationWarning
    >
      <body
        className="min-h-screen flex flex-col selection:bg-gold-dim selection:text-ink"
        suppressHydrationWarning
      >
        <Navbar />
        <main className="grow">{children}</main>

        <Footer />
      </body>
    </html>
  );
}
