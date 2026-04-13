import re

CSV_PATH = "data/merged2.txt"
CACHE_PATH = "data/.ngram_cache.pkl"

PUNCT_RE = re.compile(r"[^\w\-ʿʾ]", re.UNICODE)

LAPLACE_ALPHA = 0.3
