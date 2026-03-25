from dataclasses import dataclass
from typing import Dict, List, Tuple
import regex

from .printer import BytePrinter
from .pretokenizer import normalize

@dataclass
class ByteMergeCandidate:
    position: int
    pair: Tuple[bytes, bytes]

    def __str__(self):
        first_str = BytePrinter.format_byte_token(self.pair[0])
        second_str = BytePrinter.format_byte_token(self.pair[1])
        return f"({first_str}, {second_str}) at {self.position}"
    
    def __repr__(self) -> str:
        return f"ByteMergeCandidate(position={self.position}, pair=(0x{self.pair[0].hex()}, 0x{self.pair[1].hex()}))"
    
    @property
    def merged_token(self):
        return self.pair[0] + self.pair[1]
    
class ByteTokenSequence:
    def __init__(self, tokens: List[bytes]):
        self.tokens = tokens.copy()

    def iter_pairs(self):
        for i, pair in enumerate(zip(self.tokens[:-1], self.tokens[1:])):
            yield ByteMergeCandidate(
                position=i,
                pair=pair
            )

    def find_best_merge(self, merge_ranks: Dict[Tuple[bytes, bytes], int]):
        candidate: ByteMergeCandidate | None = None
        best_rank = float('inf')

        for cand in self.iter_pairs():
            if cand.pair in merge_ranks:
                rank = merge_ranks[cand.pair]
                if rank < best_rank:
                    candidate = cand
                    best_rank = rank

        return candidate
    
    def apply_merge(self, candidate: ByteMergeCandidate):
        self.tokens[candidate.position] = candidate.merged_token
        del self.tokens[candidate.position + 1]

    def merge_until_done(self, merge_ranks: Dict[Tuple[bytes, bytes], int]) -> List[bytes]:
        step = 0
        while True:
            candidate = self.find_best_merge(merge_ranks)
            if candidate is None:
                break

            self.apply_merge(candidate)
            step += 1

            print(f"Step {step}: merged {candidate} -> {BytePrinter.format_tokens(self.tokens)}")

        return self.tokens
    
class ByteEncoder:
    def __init__(
            self,
            merges: List[Tuple[bytes, bytes]],
            vocab: List[bytes],
            pat_str: str,
        ):
        self.pat_str = pat_str
        self._merge_ranks = {
            pair: idx
            for idx, pair in enumerate(merges)
        }
        self._token_to_id = {
            token: idx
            for idx, token in enumerate(vocab)
        }
        self._id_to_tokens = {
            idx: token
            for idx, token in enumerate(vocab)
        }

    def tokenize(self, text: str) -> List[bytes]:
        text = normalize(text)
        raw_words = regex.findall(self.pat_str, text)

        all_tokens: List[bytes] = []
        for word in raw_words:
            word_tokens = self.tokenize_word(word)
            all_tokens.extend(word_tokens)

        print(f'\nTokens: {all_tokens}')

        return all_tokens
    
    def tokenize_word(self, word: str) -> List[bytes]:
        tokens = [
            bytes([b])
            for b in word.encode('utf-8')
        ]
        byte_seq = ByteTokenSequence(tokens)
        byte_seq.merge_until_done(self._merge_ranks)
        return byte_seq.tokens
    
    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        return [
            self._token_to_id[token]
            for token in tokens
        ]
    
    def decode(self, ids: List[int]) -> str:
        tokens = [
            self._id_to_tokens[id]
            for id in ids
        ]
        return b''.join(tokens).decode('utf-8')

    