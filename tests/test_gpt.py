from pathlib import Path
import unittest
from slp3.chapter2.gpt import (
    ByteMapping,
    create_byte_mapping,
    RankBuilder,
    BytePairMerger,
    BytePairMergeLarge,
    Encoding,
)

class TestRankBuilder(unittest.TestCase):
    def test_byte_mapping(self):
        byte_mapping = create_byte_mapping()
        self.assertEqual(len(byte_mapping), 256)

        self.assertEqual(
            byte_mapping[0],
            ByteMapping(
                source=33,
                target='!'
            )
        )

        self.assertEqual(
            byte_mapping[255],
            ByteMapping(
                source=173,
                target="Ń"
            )
        )

    def test_decode_printable(self):
        builder = RankBuilder(create_byte_mapping())
        decoded = builder.decode_printable("Ġth")
        self.assertEqual(decoded, b' th')

        decoded = builder.decode_printable("\u00e7\u0137")
        self.assertEqual(decoded, b'\xe7\x95')
        

class TestBytePairMerge(unittest.TestCase):

    mock_ranks = {
        b'e': 0,
        b'n': 1,
        b'r': 2,
        b's': 3,
        b't': 4,
        b'w': 5,
        b'ne': 6,
        b'new': 7,
        b're': 8,
        b'se': 9,
        b'set': 10,
        b'renew': 11,
        b'reset': 12,
    }

    def test_small_merge(self):
        merger = BytePairMerger(b'renew', self.mock_ranks)
        print(merger.parts)
        print(merger.min_rank)

        for b in merger.iter_parts():
            print(b)

        print(merger.encode())

        merger.merge_until_done()
        print(merger.parts)
        print(merger.min_rank)

        for b in merger.iter_parts():
            print(b)

        print(merger.encode())

    def test_large_merge(self):
        merger = BytePairMergeLarge(b'renew', self.mock_ranks)

        print(merger.states)
        print(merger.heap)
        
        merger.merge_until_done()
        print(merger.states)
        print(merger.heap)

        print(merger.encode())

class TestEncoding(unittest.TestCase):
    gpt2_pattern = (
        r"""'s|'t|'re|'ve|'m|'ll|'d|[\p{Script=Han}]| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    
    def test_encoding(self):
        builder = RankBuilder(create_byte_mapping())
        vocab = builder.build(Path('./build/vocab.bpe'))

        tokenizer = Encoding(
            encoder=vocab,
            pattern=self.gpt2_pattern,
        )

        tokens = tokenizer.encode("Hello，世界。")

        decoded = tokenizer.decode(tokens)

        self.assertEqual(decoded, "Hello，世界。")