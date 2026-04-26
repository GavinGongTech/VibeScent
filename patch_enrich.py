import re

with open("src/vibescents/enrich.py", "r") as f:
    content = f.read()

# add generate_batch to EnrichmentClient protocol
content = re.sub(
    r"    def generate\(self, prompt: str\) -> EnrichmentSchemaV2:\n        \"\"\"Generate one enrichment object from a prompt.\"\"\"\n",
    """    def generate(self, prompt: str) -> EnrichmentSchemaV2:
        \"\"\"Generate one enrichment object from a prompt.\"\"\"

    def generate_batch(self, prompts: list[str]) -> list[EnrichmentSchemaV2]:
        \"\"\"Generate a batch of enrichment objects.\"\"\"
""",
    content
)

# add generate_batch to QwenOutlinesEnrichmentClient
batch_code = """
    def generate_batch(self, prompts: list[str]) -> list[EnrichmentSchemaV2]:
        full_prompts = [f"{SYSTEM_PROMPT}\\n\\n{p}" for p in prompts]
        raw_outputs = self._generator(full_prompts)
        
        results = []
        for raw in raw_outputs:
            parsed = _parse_enrichment(raw)
            if parsed is not None:
                results.append(parsed)
                continue
            repaired = _repair_payload(raw)
            parsed = _parse_enrichment(repaired)
            if parsed is not None:
                results.append(parsed)
            else:
                # If a single item fails, we still need to return something or raise?
                # For batching, better to raise or return None. But schema expects EnrichmentSchemaV2.
                # We can raise an error and the caller has to handle it, or we return an empty object.
                raise ValueError("Outlines output could not be parsed into EnrichmentSchemaV2 in batch.")
        return results
"""

content = re.sub(
    r"        raise ValueError\(\"Outlines output could not be parsed into EnrichmentSchemaV2.\"\)\n",
    '        raise ValueError("Outlines output could not be parsed into EnrichmentSchemaV2.")\n' + batch_code,
    content
)

with open("src/vibescents/enrich.py", "w") as f:
    f.write(content)

