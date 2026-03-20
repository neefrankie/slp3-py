from collections import Counter
from dataclasses import dataclass
import json
import os
import re
from typing import Dict, List, Set, Tuple



@dataclass
class MergePair:
    first: str
    second: str
    frequency: int

    def __str__(self):
        return f'({self.first} {self.second}): {self.frequency}'

    @property
    def merged_token(self):
        return self.first + self.second
    
    @property
    def pair(self):
        return (self.first, self.second)
    
    def add(self, other: 'MergePair'):
        self.frequency += other.frequency


class PairFrequencyTable:
    def __init__(self):
        self._pairs: Dict[str, MergePair] = {} # key: merged_token

    def __str__(self):
        return '\n'.join(
            f'{v}'
            for v in self._pairs.values()
        )
    
    def __len__(self):
        return len(self._pairs)

    def add(self, pair: MergePair) -> None:
        key = pair.merged_token
        if key in self._pairs:
            self._pairs[key].frequency += pair.frequency
        else:
            self._pairs[key] = pair

    def get_most_frequent(self) -> MergePair | None:
        if not self._pairs:
            return None
        return max(self._pairs.values(), key=lambda x: x.frequency)

# U+2581 (Lower One Eighth Block)
BOUNDARY_CHAR = '▁'

def split_word(word: str) -> List[str]:
    return [BOUNDARY_CHAR] + list(word) 

@dataclass
class TokenizedWord:
    text: str # 原始词，如 "lower"
    frequency: int # 在语料中出现的次数
    tokens: List[str] # 当前拆分状态，如 ['l', 'o', 'w', 'e', 'r']

    @classmethod
    def from_text(cls, word: str, freq: int) -> 'TokenizedWord':
        return cls(
            text=word,
            frequency=freq,
            tokens=split_word(word)
        )

    def __str__(self) -> str:
        return f'{self.text}: {self.frequency} {self.tokens}'

    def iter_pairs(self):
        # Should we check len(self.chars) == 1?
        # When self.chars is merged into a word,
        # the loop will be range(1-1).
        # So I don't think it's necessary.
        for i in range(len(self.tokens) - 1):
            yield MergePair(
                first=self.tokens[i],
                second=self.tokens[i+1],
                frequency=self.frequency
            )

    def apply_merge(self, pair: MergePair):
        """将 pair 合并到当前 tokens 中"""
        i = 0
        while i < len(self.tokens) - 1:
            # This modifies list in the loop,
            # so you cannot use for loop.
            if self.tokens[i] == pair.first and self.tokens[i+1] == pair.second:
                # self.tokens = self.tokens[:i] + [pair.merged_token] + self.tokens[i+2:]
                self.tokens[i] = pair.merged_token
                del self.tokens[i+1]
            else:
                i += 1

SEPARATOR_RE = re.compile(r'([,.:;?_!"()\'\s]|--)')

def basic_tokenizer(text: str) -> List[str]:
    result = SEPARATOR_RE.split(text)
    return [
        word.strip()
        for word in result
        if word.strip()
    ]

def normalize(text: str) -> str:
    """文本标准化"""
    # 1. Unicode规范化
    import unicodedata
    text = unicodedata.normalize('NFKC', text)
    
    # 2. 处理空白符
    text = ' '.join(text.split())  # 合并多个空白为单个空格
    
    # 3. 其他规范化（如大小写，可选）
    # text = text.lower()
    
    return text

class BPETokenizerTrainer:
    def __init__(
            self,
            text: str,
            specials: List[str] | None = None
        ):
        self._words: List[TokenizedWord] = self._preprocess(text)
        self._specials = specials or ['<|endoftext|>', '<|unk|>']
        self._merges: List[Tuple[str, str]] = []
        self._vocabulary: List[str] | None = None
        self._initial_vocab_size = len(self._build_initial_vocabulary()) + len(self._specials)

    def _preprocess(self, text: str) -> List[TokenizedWord]:
        """预处理：分词、添加边界标记、统计频率"""
        # 1. 标准化文本
        text = normalize(text)
        # 2. 基础分词
        raw_words = basic_tokenizer(text)

        # 3. 统计词频
        counter = Counter(raw_words)
        return [
            TokenizedWord.from_text(
                word=word,
                freq=freq,
            )
            for word, freq in counter.items()
        ]

    @property
    def merges(self) -> List[Tuple[str, str]]:
        return self._merges
    
    @property
    def vocabulary(self) -> List[str]:
        """懒加载：仅在需要时构建完整词汇表"""
        if self._vocabulary is None:
            self._vocabulary = sorted(self._build_vocabulary_from_merges())
        return self._vocabulary
    
    @property
    def current_vocab_size(self) -> int:
        return self._initial_vocab_size + len(self._merges)

    def __str__(self):
        return '\n'.join(
            str(tw)
            for tw in self._words
        )

    def _build_initial_vocabulary(self) -> Set[str]:
        """
        Example: ['e', 'n', 'r', 's', 't', 'w']
        """
        alphabet: Set[str] = set()

        for word in self._words:
            alphabet.update(word.text)

        return alphabet
    
    def _build_vocabulary_from_merges(self) -> Set[str]:
        vocab = self._build_initial_vocabulary()
        for pair in self._merges:
            vocab.add(pair[0] + pair[1])
        return vocab
    
    def _compute_pair_freqencies(self):
        freq_table = PairFrequencyTable()
        for tw in self._words:
            # What happens if all chars are merged?
            # This loop won't be executed.
            # AdjFreqCounter will never have anything related to this word.
            # If chars of every word are merged,
            # AdjFreqCounter won't hold any data.
            for pair_freq in tw.iter_pairs():
                freq_table.add(pair_freq)

        return freq_table
    
    def _apply_merge_to_all_words(self, pair: MergePair):
        for tw in self._words:
            tw.apply_merge(pair)

    def train(self, target_vocab_size: int):        
        if self.current_vocab_size >= target_vocab_size:
            return
        
        

        # What if vocab len never reached k?
        # 持续合并，直到达到目标大小或无法继续
        step = 0
        while self.current_vocab_size < target_vocab_size:
            print(f'\nStep {step}:')
            print(f'  Words before merge::')
            for tw in self._words:
                print(tw)

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
            print(f'  After merged best pair: {best_pair}')
            for tw in self._words:
                print(tw)

            self._merges.append(best_pair.pair) # 每次 append，vocab_size +1
            print(f'  Merge rules:')
            print(self._merges)
            step += 1
            print(f"Finished step {step}\n{'='*50}")


    def save(self, path: os.PathLike):
        with open(f'{path}/merges.txt', 'w') as f:
            f.write('#version: 0.1\n')
            for pair in self.merges:
                f.write(f'{pair[0]} {pair[1]}\n')

        vocab_dict = {
            token: idx
            for idx, token in enumerate(self.vocabulary)
        }
        with open(f'{path}/vocab.json', 'w') as f:
            json.dump(vocab_dict, f, indent=2)

        cfg = {
            'special_tokens': self._specials,
        }
        with open(f'{path}/tokenizer_config.json', 'w') as f:
            json.dump(cfg, f, indent=2)