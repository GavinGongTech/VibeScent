import re

with open("src/vibescents/embeddings.py", "r") as f:
    content = f.read()

# Let's find embed_multimodal_documents in Qwen3VLMultimodalEmbedder or base class
