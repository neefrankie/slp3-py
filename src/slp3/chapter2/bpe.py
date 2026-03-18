from collections import Counter
import os
import re
from typing import Dict, List, Set, Tuple

from pydantic import BaseModel, json

class MergePair(BaseModel):
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

class TokenizedWord(BaseModel):
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
        """
        Example: ['e', 'n', 're', 'se', 't', 'w']
        """
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
        while self.current_vocab_size < target_vocab_size:
            pair_freq_table = self._compute_pair_freqencies()
            best_pair = pair_freq_table.get_most_frequent()
            # Sometimes BPE training stops when:
            # best_pair.frequency < 2
            # Because merging a pair that occurs once doesn't help compression.
            if not best_pair:
                # What if we break in such case?
                print(f"Stopped early: no more pairs to merge")
                break
            print(f'Merging {best_pair}')
            self._apply_merge_to_all_words(best_pair)
            self._merges.append(best_pair.pair) # 每次 append，vocab_size +1
            # 不立即构建 full vocab（可懒加载）
            # vocab.append(best_pair.merged_token)

        # return Vocabulary(vocab)

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

class MergeCandidate(BaseModel):
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


class BPTEncoding:
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
        raw_words = basic_tokenizer(text)

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

        





if __name__ == '__main__':
    adj_ne = MergePair(first="n", second="e", frequency=2)
    adj_ne.add(MergePair(first="n", second="e", frequency=2))
    print(adj_ne)
    

    adj_ew = MergePair(first="e", second="w", frequency=4)
    adj_re = MergePair(first="r", second="e", frequency=3)
    adj_se = MergePair(first="s", second="e", frequency=2)
    adj_et = MergePair(first="e", second="t", frequency=2)
    adj_en = MergePair(first="e", second="n", frequency=2)
    adj_es = MergePair(first="e", second="s", frequency=1)

    pair_freq_table = PairFrequencyTable()
    print('\nAdjacent char frequencies:')
    pair_freq_table.add(adj_ne)
    pair_freq_table.add(adj_ew)
    pair_freq_table.add(adj_re)
    pair_freq_table.add(adj_se)
    pair_freq_table.add(adj_et)
    pair_freq_table.add(adj_en)
    pair_freq_table.add(adj_es)
    print(pair_freq_table)

    most_freq_pair = pair_freq_table.get_most_frequent()
    print('\nMost frequent pair:')
    print(most_freq_pair)

    word_renew = TokenizedWord.from_text('renew', 2)
    print('\nWord frequencies:')
    print(word_renew)

    for pair in word_renew.iter_pairs():
        print(pair)

    if most_freq_pair:
        word_renew.apply_merge(most_freq_pair)
        print('\nAfter merging n e:')
        print(word_renew)
    
    words = ['set', 'new', 'new', 'renew', 'reset', 'renew']
    text = 'set new new renew reset renew'
    trainer = BPETokenizerTrainer(text)
    print('\nWord frequencies:')
    print(trainer._words)

    vocab = trainer._build_initial_vocabulary()
    print('\nInital Vocab:')
    print(vocab)

    vocab = trainer._build_vocabulary_from_merges()
    print('\nVocab:')
    print(vocab)

    pair_freq_table = trainer._compute_pair_freqencies()
    print('\nAdjacent char frequencies:')
    print(pair_freq_table)

    trainer = BPETokenizerTrainer(text)

    trainer.train(target_vocab_size=50)
    print('\nMerges:')
    print(trainer.merges)

    print('\nTrained vocab:')
    print(trainer.vocabulary)

    bpe = BPTEncoding(
        merges=trainer.merges,
        vocab=trainer.vocabulary
    )

    tokenized = bpe.tokenize_word('renew')
    print('\nTokenized:')
    print(tokenized)

    tokens = bpe.tokenize(text)
    print('\nTokens:')
    print(tokens)

    ids = bpe.encode(text)
    print(f'\nVocab {vocab}')
    print('\nIDs:')
    print(ids)

    decoded = bpe.decode(ids)
    print(decoded)

