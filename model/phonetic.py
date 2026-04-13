from typing import NamedTuple
import re


class PhoneticMatch(NamedTuple):
    word: str
    code: str


class ChagataiPhonetic:
    """Phonetic encoder for Chagatai/Turkic languages."""

    # Phonetic groupings for Chagatai
    VOWEL_GROUPS = {
        # Front vowels
        'a': 'A', 'ä': 'A', 'å': 'A',
        'e': 'E', 'ə': 'E',
        # Back vowels
        'i': 'I', 'ı': 'I', 'y': 'I',
        'o': 'O', 
        'u': 'U', 
        'ö': 'O', 'ő': 'O', 'ø': 'O',
        'ü': 'U', 'ű': 'U',
    }

    CONSONANT_GROUPS = {
        # Labials
        'b': 'B', 'p': 'B',
        'm': 'M',
        'w': 'V', 'v': 'V',
        # Dentals/Alveolars
        'd': 'D', 't': 'D',
        'n': 'N',
        's': 'S', 'z': 'S',
        'l': 'L', 'r': 'R',
        # Velars
        'g': 'G', 'k': 'G', 'q': 'G',
        'ŋ': 'N', 'ñ': 'N',
        # Palatals
        'j': 'J', 'ž': 'J', 'c': 'J', 'č': 'J', 'ç': 'J',
        'š': 'X', 'ş': 'X', 'x': 'X',
        # Glottals
        'h': 'H',
        'ʿ': '', 'ʾ': '',  # Arabic diacritics - ignore
        'ğ': 'G', 'gh': 'G',
    }

    def __init__(self):
        self._phonetic_index: dict[str, set[str]] = {}

    def encode(self, word: str) -> str:
        """Convert word to phonetic code."""
        word = word.lower().strip()

        result = []
        prev_group = None
        skip_next = False

        for i, ch in enumerate(word):
            if skip_next:
                skip_next = False
                continue

            # Handle digraphs
            if i < len(word) - 1:
                digraph = ch + word[i + 1]
                if digraph in self.CONSONANT_GROUPS:
                    group = self.CONSONANT_GROUPS[digraph]
                    if group and group != prev_group:
                        result.append(group)
                        prev_group = group
                    skip_next = True
                    continue

            # Vowels - keep first vowel group, collapse subsequent
            if ch in self.VOWEL_GROUPS:
                group = self.VOWEL_GROUPS[ch]
                if result and result[-1] in 'AEIOU':
                    continue  # Collapse consecutive vowels
                result.append(group)
                prev_group = group
            # Consonants
            elif ch in self.CONSONANT_GROUPS:
                group = self.CONSONANT_GROUPS[ch]
                if group and group != prev_group:
                    result.append(group)
                    prev_group = group
            # Unknown - keep as-is
            else:
                if ch.isalpha():
                    result.append(ch.upper())
                    prev_group = ch.upper()

        return ''.join(result)

    def build_index(self, vocab: set[str]) -> "ChagataiPhonetic":
        """Build phonetic index from vocabulary."""
        for word in vocab:
            code = self.encode(word)
            if code not in self._phonetic_index:
                self._phonetic_index[code] = set()
            self._phonetic_index[code].add(word)
        return self

    def find_phonetic(self, word: str) -> list[PhoneticMatch]:
        """Find words with same phonetic code."""
        code = self.encode(word)
        matches = self._phonetic_index.get(code, set())
        return [PhoneticMatch(word=w, code=code) for w in sorted(matches)]

    def find_similar_phonetic(self, word: str, max_edits: int = 1) -> list[PhoneticMatch]:
        """Find words with similar phonetic codes (within edit distance)."""
        code = self.encode(word)
        results: list[PhoneticMatch] = []

        for stored_code, words in self._phonetic_index.items():
            if self._edit_distance(code, stored_code) <= max_edits:
                for w in words:
                    results.append(PhoneticMatch(word=w, code=stored_code))

        return results

    def _edit_distance(self, s1: str, s2: str) -> int:
        """Simple edit distance for phonetic codes."""
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
