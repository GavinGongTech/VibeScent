from __future__ import annotations

from pathlib import Path

from vibescents.io_utils import guess_mime_type
from vibescents.schemas import RetrievalCandidate, RerankResponse
from vibescents.settings import Settings


RERANK_SYSTEM_PROMPT = """\
You are ranking fragrance candidates for an outfit and occasion.
Score each candidate from 0.0 to 1.0.
Use the image, occasion text, candidate retrieval text, and structured metadata.
Be strict. Do not reward weak matches.
"""


class GeminiReranker:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        if not self.settings.api_key:
            raise ValueError(
                "Set GEMINI_API_KEY or GOOGLE_API_KEY before calling the Gemini API."
            )

        from google import genai

        self._client = genai.Client(api_key=self.settings.api_key)

    def rerank(
        self,
        *,
        occasion_text: str,
        candidates: list[RetrievalCandidate],
        image_path: str | Path | None = None,
    ) -> RerankResponse:
        from google.genai import types

        parts = [
            types.Part.from_text(text=self._build_prompt(occasion_text, candidates))
        ]
        if image_path is not None:
            image_file = Path(image_path)
            parts.insert(
                0,
                types.Part.from_bytes(
                    data=image_file.read_bytes(),
                    mime_type=guess_mime_type(image_file),
                ),
            )
        response = self._client.models.generate_content(
            model=self.settings.reranker_model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                system_instruction=RERANK_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=RerankResponse,
            ),
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, RerankResponse):
            return parsed
        if parsed is not None:
            return RerankResponse.model_validate(parsed)
        return RerankResponse.model_validate_json(response.text)

    @staticmethod
    def _build_prompt(occasion_text: str, candidates: list[RetrievalCandidate]) -> str:
        lines = [f"Occasion: {occasion_text}", "", "Candidates:"]
        for candidate in candidates:
            lines.append(
                "\n".join(
                    [
                        f"- fragrance_id: {candidate.fragrance_id}",
                        f"  name: {candidate.name or ''}",
                        f"  brand: {candidate.brand or ''}",
                        f"  baseline_score: {candidate.baseline_score if candidate.baseline_score is not None else ''}",
                        f"  retrieval_text: {candidate.retrieval_text}",
                        f"  metadata: {candidate.metadata}",
                    ]
                )
            )
        lines.append("")
        lines.append("Return one result per candidate fragrance_id.")
        return "\n".join(lines)
