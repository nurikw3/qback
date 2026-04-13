from collections import defaultdict
from typing import Iterable


class TrieNode:
    __slots__ = ("children", "is_word")

    def __init__(self):
        self.children: dict[str, "TrieNode"] = {}
        self.is_word: bool = False


class Trie:
    def __init__(self):
        self.root = TrieNode()
        self._size = 0

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        if not node.is_word:
            node.is_word = True
            self._size += 1

    def contains(self, word: str) -> bool:
        node = self._find_node(word)
        return node.is_word if node else False

    def starts_with(self, prefix: str) -> list[str]:
        node = self._find_node(prefix)
        if not node:
            return []
        results: list[str] = []
        self._collect(node, prefix, results)
        return results

    def _find_node(self, prefix: str) -> TrieNode | None:
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    def _collect(self, node: TrieNode, path: str, out: list[str]) -> None:
        if node.is_word:
            out.append(path)
        for ch, child in node.children.items():
            self._collect(child, path + ch, out)

    def __len__(self) -> int:
        return self._size

    def build_from_vocab(self, vocab: Iterable[str]) -> "Trie":
        for word in vocab:
            self.insert(word)
        return self
