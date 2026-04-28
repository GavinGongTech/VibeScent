import requests

resp = requests.post(
    "http://localhost:8001/search",
    json={"perfumes": ["Dior Sauvage", "Chanel Bleu de Chanel"], "budget": 200.0}
)
print(resp.json())
