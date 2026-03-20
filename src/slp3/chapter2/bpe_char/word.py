# U+2581 (Lower One Eighth Block)
from typing import List


BOUNDARY_CHAR = '▁'

def split_word(word: str) -> List[str]:
    return [BOUNDARY_CHAR] + list(word) 