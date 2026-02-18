import regex as re
import json
from collections import defaultdict, Counter
import heapq


class BPETokenizer:
    """Byte-pair encoding tokenizer. Train it on a corpus, then use
    encode/decode to convert between text and token IDs.

    The base vocabulary is the 256 raw byte values, and BPE merges
    are learned on top of that.
    """

    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}
        self.cache: dict[str, list[int]] = {}

        # GPT-2 style pre-tokenization pattern
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    # ---- training ----

    def train(self, text: str, vocab_size: int):
        num_merges = vocab_size - 256
        assert num_merges > 0, "vocab size needs to be greater than 256"

        self.vocab = {i: bytes([i]) for i in range(256)}
        word_tokens = self._build_word_freq(text)
        pair_freq = self._compute_pair_freq(word_tokens)
        heap = self._build_heap(pair_freq)

        for merge_rank in range(num_merges):
            best_pair = self._pop_best(heap)
            if best_pair is None:
                break

            new_id = 256 + merge_rank
            self.merges[best_pair] = merge_rank
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            word_tokens = self._apply_merge(word_tokens, best_pair, new_id)
            pair_freq = self._compute_pair_freq(word_tokens)
            heap = self._build_heap(pair_freq)

            if (merge_rank + 1) % 50 == 0:
                print(f"  merge {merge_rank + 1}/{num_merges}")

    def _build_word_freq(self, text: str):
        words = re.findall(self.pat, text)
        freq = Counter(words)
        return {tuple(w.encode("utf-8")): n for w, n in freq.items()}

    def _compute_pair_freq(self, word_tokens):
        counts = defaultdict(int)
        for tokens, freq in word_tokens.items():
            for i in range(len(tokens) - 1):
                counts[(tokens[i], tokens[i + 1])] += freq
        return counts

    def _build_heap(self, pair_freq):
        heap = [(-freq, pair) for pair, freq in pair_freq.items()]
        heapq.heapify(heap)
        return heap

    def _pop_best(self, heap):
        if not heap:
            return None
        freq, pair = heapq.heappop(heap)
        return None if -freq == 0 else pair

    def _apply_merge(self, word_tokens, pair, new_id):
        return {
            self._merge_pair(tokens, pair, new_id): freq
            for tokens, freq in word_tokens.items()
        }

    def _merge_pair(self, tokens, pair, new_id):
        out = []
        i = 0
        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] == pair[0]
                and tokens[i + 1] == pair[1]
            ):
                out.append(new_id)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        return tuple(out)

    # ---- encode / decode ----

    def encode(self, text: str) -> list[int]:
        ids = []
        for word in re.findall(self.pat, text):
            if word in self.cache:
                ids.extend(self.cache[word])
                continue
            tokens = list(word.encode("utf-8"))
            tokens = self._apply_bpe(tokens)
            self.cache[word] = tokens
            ids.extend(tokens)
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    def _apply_bpe(self, tokens):
        while True:
            best = self._lowest_rank_pair(tokens)
            if best is None:
                break
            tokens = list(self._merge_pair(tokens, best, 256 + self.merges[best]))
        return tokens

    def _lowest_rank_pair(self, tokens):
        best_rank, best = float("inf"), None
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair in self.merges and self.merges[pair] < best_rank:
                best_rank = self.merges[pair]
                best = pair
        return best

    # ---- persistence ----

    def save(self, path_prefix: str):
        data = {f"{a},{b}": rank for (a, b), rank in self.merges.items()}
        with open(f"{path_prefix}.merges", "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"Tokenizer saved to {path_prefix}.merges")

    def load(self, path_prefix: str):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        self.cache = {}

        # handle both "tokenizer" and "tokenizer.merges" as input
        path = (
            path_prefix if path_prefix.endswith(".merges") else f"{path_prefix}.merges"
        )
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        for pair_str, rank in sorted(raw.items(), key=lambda x: x[1]):
            a, b = map(int, pair_str.split(","))
            self.merges[(a, b)] = rank
            self.vocab[256 + rank] = self.vocab[a] + self.vocab[b]

        print(f"Tokenizer loaded with {len(self.merges)} merge rules.")
