import unittest

from slp3.chapter2.bpe_trainer import (
    MergePair,
    PairFrequencyTable,
    TokenizedWord,
    BPETrainer,
)
from slp3.chapter2.pretokenizer import (
    pre_tokenize,
)

merge_pairs = [
    MergePair(first=b"n", second=b"e", frequency=4),
    MergePair(first=b"e", second=b"w", frequency=4),
    MergePair(first=b"r", second=b"e", frequency=3),
    MergePair(first=b"s", second=b"e", frequency=2),
    MergePair(first=b"e", second=b"t", frequency=2),
    MergePair(first=b"e", second=b"n", frequency=2),
    MergePair(first=b"e", second=b"s", frequency=1)
]

text = 'set new new renew reset renew'

class TestMergePair(unittest.TestCase):
    def test_add(self):
        adj_ne = MergePair(first=b"n", second=b"e", frequency=2)
        adj_ne.add(adj_ne)

        self.assertEqual(adj_ne.frequency, 4)

        print(adj_ne)


class TestFrequencyTable(unittest.TestCase):
    
    def test_add(self):
        table = PairFrequencyTable()

        for pair in merge_pairs:
            table.add(pair)
        
        self.assertEqual(len(table), 7)

    def test_most_frequent(self):
        table = PairFrequencyTable()

        for pair in merge_pairs:
            table.add(pair)

        most_freq = table.get_most_frequent()

        self.assertIsNotNone(most_freq)
        self.assertEqual(most_freq, merge_pairs[0])

class TestTokenizedWord(unittest.TestCase):
    def test_new_word(self):
        word = TokenizedWord.from_text(
            word='Hello',
            freq=4,
        )
        print(word)  

        word = TokenizedWord.from_text(
            word="世界",
            freq=2,
        )
        print(word)

    def test_iter_pairs(self):
        word = TokenizedWord.from_text(
            word="世界",
            freq=2,
        )
        for pair in word.iter_pairs():
            print(pair)

    def test_apply_merge(self):
        word = TokenizedWord(
            text='世界',
            frequency=2,
            tokens=['世'.encode(), '界'.encode()]
        )
        print(f'\nBefore merge: {word}, {len(word.tokens)}')
        merge = MergePair(
            first='世'.encode(),
            second='界'.encode(),
            frequency=2
        )
        word.apply_merge(merge)
        print(f'After merge: {word}, {len(word.tokens)}')


class TestBPETrainer(unittest.TestCase):
    def test_words(self):
        trainer = BPETrainer(text)

        self.assertEqual(len(trainer._words), 4)
        print('\nTraining corpus')
        trainer.display_words()

        table = trainer._compute_pair_freqencies()
        print(f'Pair frequency table: {table}')

        most_freq = table.get_most_frequent()
        print(f'Most freqent pair: {most_freq}')

        trainer.train(50)


if __name__ == '__main__':
    unittest.main()