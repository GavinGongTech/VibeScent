import sys
sys.path.insert(0, "src")
from vibescents.perfume_scraper import search_perfumes
try:
    print(search_perfumes(["Bath & Body Works Sun-Washed Citrus", "Elizabeth Arden Sunflowers Sunlit Showers", "Clean Shower Fresh"], 110.0))
except Exception as e:
    import traceback
    traceback.print_exc()
