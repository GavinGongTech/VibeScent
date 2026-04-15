export default function Footer() {
  return (
    <footer className="w-full border-t border-border py-8 mt-auto">
      <div className="max-w-7xl mx-auto px-6 md:px-12 flex flex-col md:flex-row justify-between items-center gap-4">
        <span className="font-serif tracking-widest text-muted text-sm">
          VibeScent &copy; {new Date().getFullYear()}
        </span>
        <a
          href="https://github.com"
          target="_blank"
          rel="noreferrer"
          className="font-sans text-xs uppercase tracking-widest text-muted hover:text-ink transition-colors"
        >
          GitHub
        </a>
      </div>
    </footer>
  );
}
