"""Qwen3-VL-Reranker-8B pointwise reranker — local GPU, zero API keys."""

from __future__ import annotations
from pathlib import Path
from vibescents.io_utils import guess_mime_type
from vibescents.schemas import RerankResponse
from vibescents.settings import Settings

RERANK_SYSTEM_PROMPT = (
    "You are ranking fragrance candidates for an outfit and occasion. "
    "Score each candidate from 0.0 to 1.0. "
    "Use the image, occasion text, candidate retrieval text, and structured metadata. "
    "Be strict. Do not reward weak matches."
)


class Qwen3VLReranker:
    """Pointwise reranker backed by Qwen3-VL-Reranker-8B (local GPU, no API key).

    Scores each candidate using the outfit image + occasion text.
    Requires: GPU with ~16 GB VRAM. Loads alongside Qwen3-VL-Embedding-8B on 80 GB.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        self._proc = AutoProcessor.from_pretrained(
            self.settings.reranker_model, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.settings.reranker_model,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).eval()
        tok = self._proc.tokenizer
        self._yes_id = (tok.encode("yes", add_special_tokens=False) or [-1])[0]
        self._no_id = (tok.encode("no", add_special_tokens=False) or [-1])[0]

    def rerank(
        self,
        *,
        occasion_text: str,
        candidates: list,
        image_path: str | Path | None = None,
    ) -> RerankResponse:
        import base64
        import torch
        from qwen_vl_utils import process_vision_info

        img_b64, mime = None, "image/jpeg"
        if image_path is not None:
            img_bytes = Path(image_path).read_bytes()
            img_b64 = base64.b64encode(img_bytes).decode()
            mime = guess_mime_type(Path(image_path))
        results = []
        for c in candidates:
            user_parts = []
            if img_b64:
                user_parts.append(
                    {"type": "image", "image": f"data:{mime};base64,{img_b64}"}
                )
            user_parts.append(
                {
                    "type": "text",
                    "text": (
                        f"Occasion: {occasion_text}\n\n"
                        f"Fragrance: {c.retrieval_text}\n\n"
                        "Is this fragrance a strong match for the outfit and occasion?"
                    ),
                }
            )
            messages = [
                {"role": "system", "content": RERANK_SYSTEM_PROMPT},
                {"role": "user", "content": user_parts},
            ]
            text = self._proc.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            img_inputs, _ = process_vision_info(messages)
            inputs = self._proc(
                text=[text], images=img_inputs, return_tensors="pt", padding=True
            ).to("cuda")
            with torch.no_grad():
                out = self._model(**inputs)
            last = out.logits[0, -1, :]
            y = last[self._yes_id].item() if self._yes_id >= 0 else 0.0
            n = last[self._no_id].item() if self._no_id >= 0 else 0.0
            score = float(torch.softmax(torch.tensor([y, n]), dim=0)[0])
            results.append({"fragrance_id": c.fragrance_id, "score": round(score, 4)})
        results.sort(key=lambda x: x["score"], reverse=True)
        return RerankResponse.model_validate({"results": results})


GeminiReranker = Qwen3VLReranker  # backward-compat alias
