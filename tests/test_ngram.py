import unittest

from pathlib import Path

from slp3.chapter3.ngram import Ngram, iter_lines_from_file


class TestNgram(unittest.TestCase):
    file_path = Path.cwd() / 'datasets' / 'transcript.txt'

    def test_count_line(self):

        text = "okay let's see i want to go to a thai restaurant . [uh] with less than ten dollars per person"

        ngram = Ngram()
        ngram.count_line(text)

        self.assertEqual(len(ngram.seq_freq), 21)
        self.assertEqual(len(ngram.prefix_freq), 20)

    def test_load_file(self):
        line = ''
        for line in iter_lines_from_file(self.file_path):
            break

        self.assertEqual(line, "okay let's see i want to go to a thai restaurant . [uh] with less than ten dollars per person")

    def test_count(self):
        vocab = set()
        ngram = Ngram()

        for line in iter_lines_from_file(self.file_path):
            vocab.update(line.split())
            ngram.count_line(line)

        ngram.display()

        ngram.print_matrix(list(vocab)[:10])
            


if __name__ == '__main__':
    unittest.main()