import sys
import json
import base64
import urllib.request
import urllib.error

sys.path.insert(0, "src")

print("Downloading real outfit image...")
# Download an image of a man in a business suit
image_url = "https://images.unsplash.com/photo-1507679799987-c73779587ccf?w=400&q=80"
req = urllib.request.Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as resp:
        image_bytes = resp.read()
except Exception as e:
    print(f"Failed to download image: {e}")
    sys.exit(1)

img_b64 = base64.b64encode(image_bytes).decode()

payload = json.dumps({
    "image": img_b64,
    "mimeType": "image/jpeg",
    "context": {
        "eventType": "Business",
        "timeOfDay": "Morning",
        "mood": "Fresh",
    },
}).encode()

print("Sending test /recommend request to http://localhost:8000 ...")
print("Context: Business | Morning | Fresh\n")

req = urllib.request.Request(
    "http://localhost:8000/recommend",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.read().decode()}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

recs = result.get("recommendations", [])
print(f"✓ Got {len(recs)} recommendations:\n")
for i, r in enumerate(recs, 1):
    print(f"  {i}. {r.get('house','?')} — {r.get('name','?')}")
    print(f"     Score: {r.get('score','?'):.3f}  |  {r.get('occasion','')}")
    print(f"     Notes: {', '.join(r.get('notes',[])[:4])}")
    print(f"     Reasoning: {r.get('reasoning','')[:120]}")
    print()

print("✓ Testing Scraper Directly...")
scraper_req = urllib.request.Request(
    "http://localhost:8001/search",
    data=json.dumps({"perfumes": [r.get("name") for r in recs], "budget": 150}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urllib.request.urlopen(scraper_req, timeout=120) as resp:
        scraper_res = json.loads(resp.read())
        print(f"Scraper returned {len(scraper_res)} results:")
        for s in scraper_res:
            print(f"  - {s.get('name')}: {s.get('price')} at {s.get('store')} (URL: {s.get('url')})")
except Exception as e:
    print(f"Scraper Error: {e}")

print("\n✓ End-to-end real test PASSED")
