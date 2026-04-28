"""Quick end-to-end test: sends a real /recommend request and prints signal details."""
import sys, base64, json, struct, zlib, urllib.request, urllib.error

sys.path.insert(0, "src")

# ── Create a minimal valid PNG (1x1 white pixel) ─────────────────────────────
def make_png_1x1():
    def chunk(name, data):
        c = name + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw = b"\x00\xff\xff\xff"  # filter byte + RGB white
    idat = chunk(b"IDAT", zlib.compress(raw))
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend

img_b64 = base64.b64encode(make_png_1x1()).decode()

payload = json.dumps({
    "image": img_b64,
    "mimeType": "image/png",
    "context": {
        "eventType": "Gala",
        "timeOfDay": "Evening",
        "mood": "Mysterious",
    },
}).encode()

print("Sending test /recommend request to http://localhost:8000 ...")
print("Context: Gala | Evening | Mysterious\n")

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

print("✓ End-to-end test PASSED")
