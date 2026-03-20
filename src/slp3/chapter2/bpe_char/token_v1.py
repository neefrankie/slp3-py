from typing import Dict, List, Set, Tuple
import math
import heapq


class EfficientTokenSequence:
    """
    使用优先队列的高效BPE实现
    
    原理：
    1. 初始时将所有的相邻对放入优先队列（heap）
    2. 每次取出优先级最高的对
    3. 合并后，只更新受影响的相邻对（最多2个新对）
    4. 将新对加入heap，标记旧对为无效
    
    时间复杂度：从 O(n²) 降到 O(n log n)
    """
    def __init__(
            self, 
            tokens: List[str], 
            merge_ranks: Dict[Tuple[str, str], int]
        ):
        """
        Args:
            tokens: 初始token序列
            merge_ranks: 合并优先级字典（rank越小越优先）
        """

        self.tokens = tokens
        self.merge_ranks = merge_ranks

        # 优先队列，存储 (priority, position, pair)
        self.heap: List[Tuple[float, int, Tuple[str, str]]] = []

        # 优先队列，存储 (priority, position, pair)
        self.valid: Set[int] = set()
        self._build_initial_heap()

    def _get_priority(self, pair: Tuple[str, str]) -> float:
        """获取合并优先级，如果不可合并返回无限大"""
        return self.merge_ranks.get(pair, math.inf)
    
    def _build_initial_heap(self):
        """构建初始优先队列"""
        for i in range(len(self.tokens) - 1):
            pair = (self.tokens[i], self.tokens[i + 1])
            priority = self._get_priority(pair)

            if priority != math.inf:
                heapq.heappush(self.heap, (priority, i, pair))
                self.valid.add(i)

    def _is_valid(self, position: int, pair: Tuple[str, str]) -> bool:
        """检查指定位置的对是否仍然有效"""
        if position not in self.valid:
            return False
        if position >= len(self.tokens) - 1:
            return False
        return (self.tokens[position], self.tokens[position + 1]) == pair
    
    def _add_new_pair(self, position: int):
        if position < 0 or position >= len(self.tokens) - 1:
            return
        
        pair = (self.tokens[position], self.tokens[position + 1])
        priority = self._get_priority(pair)

        if priority != math.inf:
            heapq.heappush(self.heap, (priority, position, pair))
            self.valid.add(position)

    def _remove_invalid_pairs(self, position: int):
        """标记受影响的相邻对为无效"""
        # 合并位置 i 会影响：
        # - 原来的对 i (已被合并)
        # - 原来的对 i-1 (如果存在)
        # - 原来的对 i+1 (如果存在)
        for pos in [position-1, position, position+1]:
            if pos in self.valid:
                self.valid.remove(pos)

    def merge_step(self) -> bool:
        """
        Returns:
            bool: whether merged.
        """
        if not self.heap:
            return False
        
        while self.heap:
            priority, position, pair = heapq.heappop(self.heap)
            if self._is_valid(position, pair):
                # Perform merging
                merged_token = pair[0] + pair[1]
                # Invalidate affected pairs
                self._remove_invalid_pairs(position)

                # Update tokens
                self.tokens[position] = merged_token
                del self.tokens[position + 1]

                # Add new pairs
                self._add_new_pair(position-1)
                self._add_new_pair(position)

                return True
        
        return False
    
    def merge_all(self) -> List[str]:
        step = 0
        while self.merge_step():
            step += 1
            print(f"Step {step}: {self.tokens}")

        return self.tokens
    
if __name__ == "__main__":
    merge_ranks = {
        ('n', 'e'): 0,
        ('ne', 'w'): 1,
        ('r', 'e'): 2,
        ('s', 'e'): 3,
        ('se', 't'): 4,
        ('re', 'new'): 5,
        ('re', 'set'): 6
    }

    initial_tokens = ['r', 'e', 'n', 'e', 'w']

    seq = EfficientTokenSequence(initial_tokens, merge_ranks)

    final_token = seq.merge_all()

    print(final_token)
