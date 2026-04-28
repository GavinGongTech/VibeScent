import os
import requests
from dotenv import load_dotenv

load_dotenv()
params = {
    "engine": "amazon",
    "k": "Dior Sauvage cologne",
    "amazon_domain": "amazon.com",
    "api_key": os.getenv("SERPAPI_KEY"),
    "num": "1"
}
resp = requests.get("https://serpapi.com/search", params=params)
data = resp.json()
organic = data.get("organic_results", [])
if organic:
    item = organic[0]
    print("Organic keys containing image/thumb:")
    print({k: v for k, v in item.items() if "image" in k or "thumb" in k or "pic" in k})
else:
    print("No organic results")
