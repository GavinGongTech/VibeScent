"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { motion } from "framer-motion";

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <motion.header
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
      className={`fixed top-0 left-0 w-full z-50 transition-all duration-500 border-b border-transparent ${
        scrolled
          ? "bg-bg/80 backdrop-blur-md border-border py-4"
          : "bg-transparent py-6"
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 md:px-12 flex justify-between items-center">
        <Link
          href="/"
          className="font-serif text-xl tracking-widest text-ink hover:text-gold transition-colors"
        >
          VibeScent
        </Link>
        <nav className="flex gap-8 font-sans text-sm tracking-widest uppercase">
          <Link
            href="/demo"
            className="text-muted hover:text-ink transition-colors"
          >
            Demo
          </Link>
          <Link
            href="/model"
            className="text-muted hover:text-ink transition-colors"
          >
            Model
          </Link>
        </nav>
      </div>
    </motion.header>
  );
}
