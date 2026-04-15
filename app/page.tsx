import Hero from "@/components/landing/Hero";
import AboutSection from "@/components/landing/AboutSection";

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen">
      <Hero />
      <AboutSection />
    </div>
  );
}
