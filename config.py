import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "MERGED_UZB_CHAGATAI.txt"
CACHE_PATH = BASE_DIR / "data" / ".ngram_cache.pkl"

PUNCT_RE = re.compile(r"[^\w\-ʿʾ]", re.UNICODE)

LAPLACE_ALPHA = 0.3
