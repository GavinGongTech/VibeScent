import urllib.request
import json
import traceback

try:
    req = urllib.request.Request("http://127.0.0.1:8001/search", data=json.dumps({"perfumes": ["Clean Shower Fresh"], "budget": 110.0}).encode(), headers={"Content-Type": "application/json"})
    print(urllib.request.urlopen(req).read().decode())
except Exception as e:
    print("Error:", e)
    traceback.print_exc()
