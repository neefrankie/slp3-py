import re
from typing import List


SEPARATOR_RE = re.compile(r'([,.:;?_!"()\'\s]|--)')

def pre_tokenize(text: str) -> List[str]:
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