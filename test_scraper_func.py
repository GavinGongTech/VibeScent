import sys
sys.path.insert(0, "src")
from vibescents.perfume_scraper import search_perfumes
try:
    print(search_perfumes(["Dior Sauvage"], 200.0))
except Exception as e:
    import traceback
    traceback.print_exc()
