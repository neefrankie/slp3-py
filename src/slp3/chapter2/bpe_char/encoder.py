from dataclasses import dataclass
from typing import Dict, List, Tuple

from slp3.chapter2.pretokenizer import pre_tokenize, normalize
from .word import split_word, BOUNDARY_CHAR

# ===== BPE Traiing =====
@dataclass
class MergeCandidate:
    position: int
    pair: Tuple[str, str]

    def __str__(self):
        return f'({self.pair[0]} {self.pair[1]}) at {self.position}'
    
    @property
    def merged_token(self):
        return self.pair[0] + self.pair[1]
    
class TokenSequence:
    def __init__(self, tokens: List[str]):
        self.tokens = tokens.copy()

    def iter_pairs(self):
        for i in range(len(self.tokens) - 1):
            yield MergeCandidate(
                position=i,
                pair=(self.tokens[i], self.tokens[i+1])
            )

    def find_best_merge(self, merge_ranks: Dict[Tuple[str, str], int]) -> MergeCandidate | None:
        """
        Find adjacent pairs with highest precedence.
        """
        candidate: MergeCandidate | None = None
        best_rank = float('inf')
        # For a sequence of len n to be completely merged into a sigle word,
        # it has to iterate n - 1 + ... 2 times.
        for cand in self.iter_pairs():
            if cand.pair in merge_ranks:
                rank = merge_ranks[cand.pair]
                if rank < best_rank:
                    best_rank = rank
                    candidate = cand

        return candidate
    
    def apply_merge(self, candidate: MergeCandidate) -> None:
        """
        Merge adjcent pairs with highest precedence.
        """
        self.tokens[candidate.position] = candidate.merged_token
        del self.tokens[candidate.position + 1]

    def merge_until_done(
            self,
            merge_ranks: Dict[Tuple[str, str], int]
        ) -> List[str]:
        """持续合并直到无法继续"""
        step = 0
        while True:
            candidate = self.find_best_merge(merge_ranks)
            if candidate is None:
                break

            self.apply_merge(candidate)
            step += 1
            print(f'Step {step}: merged {candidate.pair} -> {candidate.merged_token}')

        return self.tokens


class BPEEncoder:
    EOF = '<|endoftext|>'
    UNK = '<|unk|>'

    def __init__(
            self,
            merges: List[Tuple[str, str]],
            vocab: List[str],
        ):
        # 构建合并优先级字典
        # 越早合并的pair优先级越高（rank越小）
        self._merge_ranks = {
            pair: idx
            for idx, pair in enumerate(merges)
        }
        self._token_to_id = {
            token: idx
            for idx, token in enumerate(vocab)
        }
        self.unk_id = len(self._token_to_id)
        self._token_to_id[self.UNK] = self.unk_id
        self._id_to_tokens = {
            idx: token
            for idx, token in enumerate(vocab)
        }
        self._id_to_tokens[self.unk_id] = self.UNK

    def tokenize_word(self, word: str) -> List[str]:
        """
        分词一个单词
        使用TokenSequence封装
        """
        # 1. 初始化token序列
        initial_tokens = split_word(word)
        sequence = TokenSequence(initial_tokens)

        print(f'\nTokenizing: {initial_tokens}')

        # 2. 持续合并直到无法继续
        final_tokens = sequence.merge_until_done(self._merge_ranks)

        print(f'Finished: {final_tokens}')

        return final_tokens
    
    def tokenize(self, text: str) -> List[str]:
        # 1. 标准化文本
        text = normalize(text)
        # 2. 基础分词
        raw_words = pre_tokenize(text)

        all_tokens: List[str] = []

        for word in raw_words:
            word_tokens = self.tokenize_word(word)
            all_tokens.extend(word_tokens)

        print(f'\nTokens: {all_tokens}')

        return all_tokens
    
    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        return [
            self._token_to_id.get(t, self.unk_id)
            for t in tokens
        ]
    
    def decode(self, ids: List[int]) -> str:
        words = [
            self._id_to_tokens.get(id_, '')
            for id_ in ids
        ]
        text = ''.join(words)
        text = text.replace(BOUNDARY_CHAR, ' ')
        return text.strip()