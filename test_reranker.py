"""Direct test of the Gemma-3-27B reranker on Nvidia NIM.

Uses the 3 fragrances returned by test_real.py (Business | Morning | Fresh)
and re-ranks them alongside a deliberately bad candidate so we can see the
model actively sorting them.
"""
import sys
import time
sys.path.insert(0, "src")

from dotenv import load_dotenv
load_dotenv()

from vibescents.reranker import Qwen3VLReranker
from vibescents.schemas import RetrievalCandidate

OCCASION = "Business meeting in the morning, fresh and professional vibe"

CANDIDATES = [
    RetrievalCandidate(
        fragrance_id="1",
        retrieval_text=(
            "Brand: Clean Reserve | Name: H2Eau Golden Citrus | "
            "Accords: Aquatic, Fresh, Citrus | "
            "Top: Bergamot, Mandarin Orange | Heart: Jasmine | Base: Musk | "
            "Formality: 0.7 | Season: Summer | Gender: Neutral"
        ),
    ),
    RetrievalCandidate(
        fragrance_id="2",
        retrieval_text=(
            "Brand: Bath & Body Works | Name: White Citrus for Men | "
            "Accords: Citrus, Fresh, Clean | "
            "Top: Bergamot, Grapefruit, Lemon Zest | Heart: Mandarin | Base: Musk | "
            "Formality: 0.6 | Season: Spring | Gender: Male"
        ),
    ),
    RetrievalCandidate(
        fragrance_id="3",
        retrieval_text=(
            "Brand: Nikos | Name: Sculpture Ocean Light | "
            "Accords: Aquatic, Fresh, Woody | "
            "Top: Bergamot, Green Tea | Heart: Cypress | Base: Cedar, Musk | "
            "Formality: 0.5 | Season: Summer | Gender: Male"
        ),
    ),
    RetrievalCandidate(
        fragrance_id="4",
        retrieval_text=(
            "Brand: Versace | Name: Crystal Noir | "
            "Accords: Floral, Oriental, Sweet, Heavy | "
            "Top: Ginger, Cardamom | Heart: Peony, Gardenia | Base: Amber, Musk, Sandalwood | "
            "Formality: 0.9 | Season: Winter | Gender: Female"
        ),
    ),
]

print("=" * 60)
print("Reranker: google/gemma-3-27b-it @ Nvidia NIM")
print(f"Occasion: {OCCASION}")
print(f"Candidates: {len(CANDIDATES)}")
print("=" * 60)

reranker = Qwen3VLReranker()
t0 = time.time()
resp = reranker.rerank(occasion_text=OCCASION, candidates=CANDIDATES)
elapsed = time.time() - t0

print(f"\nRerank completed in {elapsed:.1f}s\n")
print("Ranked results (best → worst match):")
for rank, r in enumerate(resp.results, 1):
    cand = next(c for c in CANDIDATES if c.fragrance_id == r.fragrance_id)
    name_line = cand.retrieval_text.split("|")[1].strip() if "|" in cand.retrieval_text else cand.retrieval_text
    print(f"  {rank}. [{r.fragrance_id}] {name_line}")
    print(f"       Overall: {r.overall_score:.2f}  |  Formality: {r.formality_score:.2f}  |  Season: {r.season_score:.2f}  |  Freshness: {r.freshness_score:.2f}")
    print(f"       Gemma says: {r.explanation}")
    print()
