import unittest

from slp3.chapter2.bpe_char import (
    MergePair,
    PairFrequencyTable,
    BPETokenizerTrainer,
    TokenizedWord,
    BPEEncoder
)

merge_pairs = [
    MergePair(first="e", second="w", frequency=4),
    MergePair(first="r", second="e", frequency=3),
    MergePair(first="s", second="e", frequency=2),
    MergePair(first="e", second="t", frequency=2),
    MergePair(first="e", second="n", frequency=2),
    MergePair(first="e", second="s", frequency=1),
]

text = 'set new new renew reset renew'

class TestMergePair(unittest.TestCase):
    def test_add(self):
        adj_ne = MergePair(first="n", second="e", frequency=2)
        adj_ne.add(MergePair(first="n", second="e", frequency=2))
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

class TesTokenizedWord(unittest.TestCase):
    def test_new_tokenized_word(self):
        word = TokenizedWord.from_text('renew', 2)
        self.assertEqual(word.text, 'renew')
        self.assertEqual(word.frequency, 2)

    def test_iter_pairs(self):
        word = TokenizedWord.from_text('renew', 2)
        pairs = [
            MergePair(first='r', second='e', frequency=2),
            MergePair(first='e', second='n', frequency=2),
            MergePair(first='n', second='e', frequency=2),
            MergePair(first='e', second='w', frequency=2)
        ]
        i = 0
        for pair in word.iter_pairs():
            self.assertEqual(pair, pairs[i])

    def test_apply_merge(self):
        word = TokenizedWord.from_text('renew', 2)
        word.apply_merge(MergePair(first='n', second='e', frequency=2))
        self.assertEqual(word.tokens, ['r', 'e', 'ne', 'w'])

class TestBPETrainer(unittest.TestCase):

    def test_word_frequency(self):
        trainer = BPETokenizerTrainer(text)
        self.assertEqual(len(trainer._words), 4)

    def test_initial_vocab(self):
        trainer = BPETokenizerTrainer(text)
        vocab = trainer._build_initial_vocabulary()
        print(vocab)

    def test_most_frequent(self):
        trainer = BPETokenizerTrainer(text)
        most_freq = trainer._compute_pair_freqencies().get_most_frequent()
        print(most_freq)

    def test_train(self):
        trainer = BPETokenizerTrainer(text)

        trainer.train(target_vocab_size=50)
        print('\nMerges:')
        print(trainer.merges)

        print('\nTrained vocab:')
        print(trainer.vocabulary)


class TestBPEEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._trainer = BPETokenizerTrainer(text)
        cls._trainer.train(target_vocab_size=50)

    def test_tokenizer_word(self):
        tokenized = BPEEncoder(
            self._trainer.merges,
            self._trainer.vocabulary,
        )

        tokenized = tokenized.tokenize_word('renew')
        print(tokenized)

    def test_encode(self):
        encoder = BPEEncoder(
            merges=self._trainer.merges,
            vocab=self._trainer.vocabulary
        )

        ids = encoder.encode(text)
        print('\nIDs:')
        print(ids)

        decoded = encoder.decode(ids)
        print(decoded)



if __name__ == '__main__':
    unittest.main()

