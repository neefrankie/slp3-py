from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


class Ngram:
    def __init__(self, n=2):
        self.n = n
        self.seq_freq: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.prefix_freq: Dict[Tuple[str, ...], int] = defaultdict(int)

    def count_line(self, line: str):
        words = ['<s>'] * (self.n-1) + line.split() + ['</s>']
        # How to determine the index?
        # Suppose we have the bigram sequence:
        # <s> okay let's see i want to go to </s>
        # 0   1    2     3   4  5   6  7   8  9
        # To get the first pair, start from index 0 and the 
        # slice will be [0:2].
        for i in range(len(words) - self.n + 1):
            key = words[i:i+self.n]
            self.seq_freq[tuple(key)] += 1
            self.prefix_freq[tuple(key[:-1])] += 1

    def display(self):
        print(f'Total sequence frequency: {len(self.seq_freq)}')
        print(f'Total suffix frequency: {len(self.prefix_freq)}')

    def print_matrix(self, vocab: List[str]):
        header_col_width = 10
        col_width = 7
        
        header = ''.center(header_col_width) + ' | ' + 'SUFFIX'.center(col_width) + ' | '
        
        for wrod in vocab:
            header += f'{wrod}'.center(col_width) + ' | '
        
        print(header)
        print('-' * len(header))

        for prefix, freq in self.prefix_freq.items():
            suf_str = ' '.join(prefix)
            row_str = f"{suf_str}".center(header_col_width) + " | "
            row_str += f"{freq}".center(col_width) + " | "

            for word in vocab:
                key = prefix + (word,)
                joined_freq = self.seq_freq[key]
                row_str += f"{joined_freq}".center(col_width) + " | "

            print(row_str)

# Data downloaded from https://github.com/wooters/berp-trans/blob/master/transcript.txt
def iter_lines_from_file(file_path: Path):
    with file_path.open(encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            idx = line.find(' ')
            line = line[idx+1:]
            yield line.strip()




