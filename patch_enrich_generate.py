import re

with open("src/vibescents/enrich.py", "r") as f:
    content = f.read()

# Change _build_outlines_generator return
content = re.sub(
    r"return outlines.generate.json\(model, EnrichmentSchemaV2\), None",
    "return model",
    content
)
content = re.sub(
    r"return outlines.generate.json\(model, EnrichmentSchemaV2\), tokenizer",
    "return model",
    content
)

# Update generate method
content = re.sub(
    r"raw = self._generator\(f\"\{SYSTEM_PROMPT\}\\n\\n\{prompt\}\"\)",
    "raw = self._generator(f\"{SYSTEM_PROMPT}\\n\\n{prompt}\", EnrichmentSchemaV2)",
    content
)

# Update generate_batch method
content = re.sub(
    r"raw_outputs = self._generator\(full_prompts\)",
    "raw_outputs = self._generator.batch(full_prompts, EnrichmentSchemaV2) if hasattr(self._generator, 'batch') else [self._generator(p, EnrichmentSchemaV2) for p in full_prompts]",
    content
)

with open("src/vibescents/enrich.py", "w") as f:
    f.write(content)

print("enrich.py patched")
