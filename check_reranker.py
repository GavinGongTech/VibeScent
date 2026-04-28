import sys
sys.path.insert(0, "src")
from vibescents.reranker import Qwen3VLReranker
r = Qwen3VLReranker()
print("Model:", r.model)
print("Base URL:", r.client.base_url)
print("API key set:", bool(r.api_key))
