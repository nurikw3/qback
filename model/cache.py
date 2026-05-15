import os
import pickle
from pathlib import Path

from data.loader import load_corpus
from model.ngram import NgramModel


MODEL_SOURCE_FILES = (
    Path(__file__),
    Path(__file__).with_name("ngram.py"),
    Path(__file__).with_name("trie.py"),
    Path(__file__).with_name("fuzzy.py"),
    Path(__file__).with_name("phonetic.py"),
)


def _is_cache_fresh(csv_path: os.PathLike | str, cache_path: os.PathLike | str) -> bool:
    cache_mtime = os.path.getmtime(cache_path)
    if cache_mtime <= os.path.getmtime(csv_path):
        return False

    return all(cache_mtime > path.stat().st_mtime for path in MODEL_SOURCE_FILES)


def load_or_build_model(csv_path, cache_path):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path) and _is_cache_fresh(csv_path, cache_path):
        try:
            with open(cache_path, "rb") as f:
                model = pickle.load(f)
            model.ensure_runtime_indexes()
            return model
        except (AttributeError, EOFError, pickle.PickleError):
            pass

    sentences = load_corpus(csv_path)
    model = NgramModel(sentences)

    with open(cache_path, "wb") as f:
        pickle.dump(model, f)

    return model
