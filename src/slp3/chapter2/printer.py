from typing import List


class BytePrinter:
    SPECIAL_DISPLAY = {
        '▁'.encode(): '▁',  # 边界标记
        b' ': '␣',   # 空格
        b'\t': '→',  # 制表符
        b'\n': '↵',  # 换行
    }

    @classmethod
    def format_byte_token(cls, token: bytes) -> str:
        if token in cls.SPECIAL_DISPLAY:
            return cls.SPECIAL_DISPLAY[token]
        
        try:
            decoded = token.decode('utf-8')
            # 可打印字符就是在 Unicode 字符数据库 (参见 unicodedata) 中分组为主类别
            # Letter, Mark, Number, Punctuation 或 Symbol (L, M, N, P 或 S) 的字符；
            # 加上 ASCII 空格符 0x20。
            # 不可打印字符就是分组为 Separator 或 Other (Z 或 C) 的字符，ASCII 空格符除外。
            # See https://docs.python.org/zh-cn/3.14/library/stdtypes.html#str.isprintable
            # For example, '世界'.encode().isprintable() == True
            if all(c.isprintable() for c in decoded):
                return decoded
        except:
            pass

        return f"0x{token.hex()}"
    
    @classmethod
    def format_pair(cls, first: bytes, second: bytes) -> str:
        return f"({cls.format_byte_token(first), cls.format_byte_token(second)})"
    
    @classmethod
    def format_tokens(cls, tokens: List[bytes]) -> str:
        formatted = [cls.format_byte_token(token) for token in tokens]
        return '[' + ', '.join(formatted) + ']'