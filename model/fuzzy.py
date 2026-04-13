from typing import NamedTuple


class FuzzyMatch(NamedTuple):
    word: str
    distance: int


def levenshtein(s1: str, s2: str) -> int:
    """Fast Levenshtein distance."""
    if s1 == s2:
        return 0
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    row = list(range(len(s1) + 1))
    for i, c2 in enumerate(s2):
        prev = i
        row[0] = i + 1
        for j, c1 in enumerate(s1):
            nxt = row[j + 1]
            row[j + 1] = prev if c1 == c2 else min(prev, nxt, row[j]) + 1
            prev = nxt
    return row[-1]


class FuzzyMatcher:
    """Fuzzy matcher using n-gram index for fast candidate selection."""

    def __init__(self):
        self._vocab: list[str] = []
        self._ngram_index: dict[str, set[int]] = {}

    def build_from_vocab(self, vocab: set[str]) -> "FuzzyMatcher":
        self._vocab = sorted(vocab)
        for idx, word in enumerate(self._vocab):
            for ngram in self._ngrams(word):
                if ngram not in self._ngram_index:
                    self._ngram_index[ngram] = set()
                self._ngram_index[ngram].add(idx)
        return self

    def _ngrams(self, word: str, n: int = 2) -> set[str]:
        """Generate character n-grams."""
        padded = f" {word} "
        return {padded[i:i+n] for i in range(len(padded) - n + 1)}

    def find_similar(self, word: str, max_dist: int = 2, limit: int = 10) -> list[FuzzyMatch]:
        """Find similar words using n-gram overlap + Levenshtein."""
        if not word:
            return []

        word_ngrams = self._ngrams(word)
        candidates: dict[int, int] = {}

        for ngram in word_ngrams:
            if ngram in self._ngram_index:
                for idx in self._ngram_index[ngram]:
                    candidates[idx] = candidates.get(idx, 0) + 1

        scored: list[tuple[int, int]] = sorted(candidates.items(), key=lambda x: -x[1])
        top_indices = [idx for idx, _ in scored[:100]]

        results: list[FuzzyMatch] = []
        for idx in top_indices:
            candidate = self._vocab[idx]
            dist = levenshtein(word, candidate)
            if dist <= max_dist:
                results.append(FuzzyMatch(word=candidate, distance=dist))
            if len(results) >= limit * 2:
                break

        results.sort(key=lambda m: (m.distance, m.word))
        return results[:limit]

    def __len__(self) -> int:
        return len(self._vocab)
