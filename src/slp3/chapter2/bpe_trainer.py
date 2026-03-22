from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple
import regex

from .pretokenizer import normalize
from .printer import BytePrinter


@dataclass
class MergePair:
    first: bytes
    second: bytes
    frequency: int

    def __str__(self):
        first_str = BytePrinter.format_byte_token(self.first)
        second_str = BytePrinter.format_byte_token(self.second)
        return f"({first_str}, {second_str}): {self.frequency}"
    
    def __repr__(self) -> str:
        return f"ByteMergePair(first=0x{self.first.hex()}, second=0x{self.second.hex()}, frequency={self.frequency})"
    
    @property
    def merged_token(self) -> bytes:
        return self.first + self.second
    
    @property
    def pair(self):
        return (self.first, self.second)
    
    def add(self, other: 'MergePair'):
        self.frequency += other.frequency

class PairFrequencyTable:

    def __init__(self):
        self._pairs: Dict[bytes, MergePair] = {}

    def __len__(self):
        return len(self._pairs)
    
    def __str__(self):
        return '\n'.join(
            str(pair)
            for pair in self._pairs.values()
        )

    def add(self, pair: MergePair) -> None:
        key = pair.merged_token
        if key in self._pairs:
            self._pairs[key].add(pair)
        else:
            self._pairs[key] = pair

    def get_most_frequent(self) -> MergePair | None:
        # The table will empty when all words are merged.
        # max() on empty table will raise ValueError
        if not self._pairs:
            return None
        return max(self._pairs.values(), key=lambda pair: pair.frequency)



@dataclass
class TokenizedWord:
    text: str
    frequency: int
    tokens: List[bytes]

    @classmethod
    def from_text(cls, word: str, freq: int) -> 'TokenizedWord':
        return cls(
            text=word,
            frequency=freq,
            tokens=[
                bytes([b])
                for b in word.encode('utf-8')
            ]
        )

    def __str__(self):
        tokens_str = BytePrinter.format_tokens(self.tokens)
        return f"'{self.text}' (freq={self.frequency}): {tokens_str}"
    
    def __repr__(self):
        return f"ByteTokenizedWord(text='{self.text}', tokens={self.tokens})"
    
    def iter_pairs(self):
        for i in range(len(self.tokens) - 1):
            yield MergePair(
                first=self.tokens[i],
                second=self.tokens[i + 1],
                frequency=self.frequency
            )

    def apply_merge(self, pair: MergePair):
        i = 0
        while i < len(self.tokens) - 1:
            if self.tokens[i] == pair.first and self.tokens[i + 1] == pair.second:
                self.tokens[i] = pair.merged_token
                self.tokens.pop(i + 1)
            else:
                i += 1

class BPETrainer:
    def __init__(
            self,
            text: str,
            pat_str: str
    ):
        self.pat_str = pat_str
        self.merges: List[Tuple[bytes, bytes]] = []
        self.vocab = {
            bytes([i])
            for i in range(256)
        }

        self._words: List[TokenizedWord] = self._preprocess(text)

    @property
    def vocabulary(self) -> List[str]:
        return [
            BytePrinter.format_byte_token(v)
            for v in sorted(self.vocab)
        ]
        

    def display_words(self):
        for w in self._words:
            print(w)

    def _preprocess(self, text: str) -> List[TokenizedWord]:
        """预处理：分词、添加边界标记、统计频率"""
        # 1. 标准化文本
        text = normalize(text)

        # 2. 基础分词
        raw_words = regex.findall(self.pat_str, text)

        # 3. 统计词频
        counter = Counter(raw_words)
        return [
            TokenizedWord.from_text(
                word=word,
                freq=freq,
            )
            for word, freq in counter.items()
        ]
    
    def _compute_pair_freqencies(self):
        freq_table = PairFrequencyTable()
        # If all words are merged, the frequency table will be empty.
        for tw in self._words:
            for pair_freq in tw.iter_pairs():
                freq_table.add(pair_freq)

        return freq_table
    
    def _apply_merge_to_all_words(self, pair: MergePair):
        for tw in self._words:
            tw.apply_merge(pair)

    def train(self, target_vocab_size: int):        
        if len(self.vocab) >= target_vocab_size:
            return

        step = 0
        while len(self.vocab) < target_vocab_size:
            print(f'\nStep {step}:')
            print(f'  Words before merge::')
            self.display_words()

            pair_freq_table = self._compute_pair_freqencies()
            print(f'  Pair frequencies:')
            print(pair_freq_table)

            best_pair = pair_freq_table.get_most_frequent()
            # Sometimes BPE training stops when:
            # best_pair.frequency < 2
            # Because merging a pair that occurs once doesn't help compression.
            if not best_pair:
                # What if we break in such case?
                print(f"Stopped early: no more pairs to merge")
                break
            self._apply_merge_to_all_words(best_pair)
            # If we use a single global frequency table,
            # what should we do?
            # After the best_pair is applied, all pairs 
            # starting with best_pair.second or ending with best_pair.first sould be removed.
            # And the _apply_merge_to_all_words() might
            # have to return pairs adjacent to best_pair.
            print(f'  After merged best pair: {best_pair}')
            self.display_words()

            self.merges.append(best_pair.pair) # 每次 append，vocab_size +1
            self.vocab.add(best_pair.merged_token)
            
            print(f'  Merge rules:')
            print(self.merges)
            step += 1
            print(f"Finished step {step}\n{'='*50}")
