# Adapted from https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

@dataclass
class ByteMapping:
    """
    Maps a byte to a printable character.
    source: any of the first 256 unicode code points.
    target: printable character from unicode table.
    """
    source: int # 0 - 255
    target: str

    def __str__(self):
        return f"{self.source} -> {self.target}({ord(self.target)})"

def create_byte_mapping() -> List[ByteMapping]:
    # Filter those not printable and space.
    # The result contains 188 characters:
    # 33 - 126
    # 161 - 172
    # 174 - 255
    byte_mapping: List[ByteMapping] = []
    non_printable: List[int] = []
    for b in range(256):
        if chr(b).isprintable() and chr(b) != " ":
            byte_mapping.append(
                ByteMapping(
                    source=b,
                    target=chr(b)
                )
            )
        else:
            non_printable.append(b)

    for n, b in enumerate(non_printable):
        # Use character from 256 until 323 to represent non-printable bytes.
        # Last one is Ń (173)
        byte_mapping.append(
            ByteMapping(
                source=b,
                target=chr(256 + n)
            )
        )
    assert len(byte_mapping) == 256
    return byte_mapping

# You can download the files from:
# Merge file: https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe
# Dictionary: https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json
def load_merges(path: Path):
    with path.open() as f:
        contents = f.read()
        lines = contents.split("\n")[1:-1]
        merges = [
            split_merge_line(line)
            for line in lines
        ]

        return merges
    
def split_merge_line(line: str) -> Tuple[str, str]:
    # line is space separated.
    first, second = line.split()
    return first, second


class RankBuilder:
    def __init__(self, byte_mapping: List[ByteMapping]):
        self.byte_mapping = byte_mapping
        # Printable characters to the actual unicode code point.
        self.printable_to_byte: Dict[str, int] = {
            m.target: m.source
            for m in byte_mapping
        }

    def decode_printable(self, text: str) -> bytes:
        # Convert printable characters to their real byte representation.
        return bytes([self.printable_to_byte[c] for c in text])
    
    
    def build(
            self,
            merge_file_path: Path,
        ) -> Dict[bytes, int]:

        merges = load_merges(merge_file_path)

        bpe_ranks: Dict[bytes, int] = {
            bytes([m.source]): i
            for i, m in enumerate(self.byte_mapping)
        }

        n = len(bpe_ranks)
        for first, second in merges:
            first_byte = self.decode_printable(first)
            second_byte = self.decode_printable(second)
            bpe_ranks[first_byte + second_byte] = n
            n += 1

        return bpe_ranks
    