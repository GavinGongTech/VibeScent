"""Nvidia NIM API pointwise reranker — async batching, zero local VRAM."""

from __future__ import annotations
import asyncio
import base64
import logging
import os
from pathlib import Path
from openai import AsyncOpenAI

from vibescents.io_utils import guess_mime_type
from vibescents.schemas import RerankResponse
from vibescents.settings import Settings

logger = logging.getLogger(__name__)

RERANK_SYSTEM_PROMPT = (
    "You are an expert fragrance and fashion judge. "
    "You will be given an occasion, a fragrance profile, and optionally an outfit image. "
    "Is this fragrance a strong match for the outfit and occasion?\n"
    "Reply strictly in this format:\n"
    "Verdict: YES or NO\n"
    "Reason: <One short, punchy sentence explaining why>"
)

class NvidiaNIMReranker:
    """Pointwise reranker backed by Nvidia NIM API (build.nvidia.com).

    Sends concurrent asynchronous requests to leverage Nvidia's Dynamic Continuous Batching,
    returning results without consuming any local GPU VRAM.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.api_key = os.environ.get("NVIDIA_API_KEY", "")
        if not self.api_key:
            logger.warning("NVIDIA_API_KEY not set in environment. API reranking will fail.")
        
        # Meta's Llama 4 Maverick or Llama 3.2 90B are excellent zero-shot vision judges
        self.model = "google/gemma-3-27b-it"
        
        self.client = AsyncOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key,
            max_retries=1,
            timeout=15.0,   # 15s per request — fail fast, don't block the whole pipeline
        )

    def rerank(
        self,
        *,
        occasion_text: str,
        candidates: list,
        image_path: str | Path | None = None,
    ) -> RerankResponse:
        """Synchronous wrapper around the async batching logic."""
        # engine.py calls this synchronously. We spin up an event loop to handle the batch.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already inside an async context (e.g. FastAPI) — run in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run,
                                     self._async_rerank_batch(occasion_text, candidates, image_path))
                return future.result()
            
        return asyncio.run(self._async_rerank_batch(occasion_text, candidates, image_path))

    async def _async_rerank_batch(
        self, occasion_text: str, candidates: list, image_path: str | Path | None
    ) -> RerankResponse:
        img_b64, mime = None, "image/jpeg"
        if image_path is not None:
            img_bytes = Path(image_path).read_bytes()
            img_b64 = base64.b64encode(img_bytes).decode()
            mime = guess_mime_type(Path(image_path))

        async def score_single(c) -> dict:
            user_parts = []
            if img_b64:
                user_parts.append(
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}
                )
            
            user_parts.append(
                {
                    "type": "text",
                    "text": (
                        f"Occasion: {occasion_text}\n\n"
                        f"Fragrance Candidate: {c.retrieval_text}\n\n"
                        "Is this fragrance a strong match for the outfit and occasion?"
                    ),
                }
            )

            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": RERANK_SYSTEM_PROMPT},
                        {"role": "user", "content": user_parts},
                    ],
                    max_tokens=60,
                    temperature=0.2,
                )
                text_out = response.choices[0].message.content.strip()
                
                # Parse "Verdict: YES\nReason: Blah"
                lines = [line.strip() for line in text_out.split('\n') if line.strip()]
                
                score = 0.1
                explanation = "A distinctive fragrance selected for your look."
                
                for line in lines:
                    if line.lower().startswith("verdict:"):
                        score = 0.9 if "yes" in line.lower() else 0.1
                    elif line.lower().startswith("reason:"):
                        explanation = line.split(":", 1)[1].strip()

            except Exception as e:
                logger.warning("NIM API call failed for candidate %s: %s", c.fragrance_id, e)
                score = 0.5
                explanation = "Scored via fallback due to API timeout."

            return {
                "fragrance_id": c.fragrance_id,
                "overall_score": score,
                "formality_score": score,
                "season_score": score,
                "freshness_score": score,
                "explanation": explanation,
            }

        # Fire all requests concurrently! Nvidia's vLLM backend will dynamically batch them.
        tasks = [score_single(c) for c in candidates]
        results = await asyncio.gather(*tasks)

        # Sort highest score first
        results.sort(key=lambda x: x["overall_score"], reverse=True)
        return RerankResponse.model_validate({"results": results})


# Alias to allow engine.py to import it without modifications
Qwen3VLReranker = NvidiaNIMReranker
GeminiReranker = NvidiaNIMReranker
