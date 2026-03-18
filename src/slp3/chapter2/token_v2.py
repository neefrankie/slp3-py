import heapq
from typing import List, Tuple, Dict, Optional, Set

class EfficientTokenizer:
    """
    高效的BPE合并实现
    """
    def __init__(self, tokens: List[str], merge_ranks: Dict[Tuple[str, str], int]):
        """
        Args:
            tokens: 初始token序列，如 ['r', 'e', 'n', 'e', 'w']
            merge_ranks: 合并优先级字典，rank越小越优先合并
        """
        self.tokens = tokens
        self.merge_ranks = merge_ranks

        # 优先队列，存储 (rank, position)
        # rank越小越优先，position用于唯一标识
        self.heap: List[Tuple[int, int]] = []

        # 记录哪些位置的对是有效的
        # 初始时所有位置都有效
        self.valid_positions: Set[int] = set()

        # 构建初始堆
        self._build_initial_heap()

    def _build_initial_heap(self):
        """构建初始堆"""
        for i in range(len(self.tokens) - 1):
            pair = (self.tokens[i], self.tokens[i + 1])
            if pair in self.merge_ranks:
                rank = self.merge_ranks[pair]
                heapq.heappush(self.heap, (rank, i))
                self.valid_positions.add(i)
                print(f'Initialized: position={i}, pair={pair}, rank={rank}')

    def _is_valid(self, position: int) -> bool:
        """
        检查指定位置的对是否仍然有效
        需要满足：
        1. 位置在valid_positions中
        2. 位置仍然在token序列范围内
        3. 当前位置的pair和之前记录的一致（由调用者保证）
        """
        if position not in self.valid_positions:
            return False
        if position >= len(self.tokens) - 1:
            return False
        return True
    
    def get_best_pair(self) -> Tuple[int, Tuple[str, str], int] | None:
        """
        获取当前最佳的合并对
        对应"n e has priority 0, so it's the first merge"
        """
        while self.heap:
            rank, position = heapq.heappop(self.heap)

            if self._is_valid(position):
                pair = (self.tokens[position], self.tokens[position + 1])
                print(f'Best pair: position={position}, pair={pair}, rank={rank}')
                return position, pair, rank
            
        return None
    
    def _invalidate_affected_pairs(self, merge_position: int):
        """
        标记受影响的pair为无效
        
        合并位置i会影响：
        - 位置i本身（被合并了）
        - 位置i-1（左边邻居）
        - 位置i+1（右边邻居）
        """
        affected = [merge_position - 1, merge_position, merge_position + 1]
        for pos in affected:
            if pos in self.valid_positions:
                self.valid_positions.remove(pos)

    def _add_new_pairs(self, merge_position: int):
        """
        添加新产生的pair
        
        合并位置i后，会产生两个新pair：
        - 位置i-1: (token[i-1], merged_token)
        - 位置i: (merged_token, token[i+2])
        """
        # 添加左边的pair
        left_pos = merge_position - 1
        if left_pos >= 0:
            pair = (self.tokens[left_pos], self.tokens[merge_position])
            if pair in self.merge_ranks:
                rank = self.merge_ranks[pair]
                heapq.heappush(self.heap, (rank, left_pos))
                self.valid_positions.add(left_pos)
                print(f'  Added: position={left_pos}, pair={pair}, rank={rank}')

        # 添加右边的pair（即合并位置本身）
        right_pos = merge_position
        if right_pos < len(self.tokens) - 1:
            pair = (self.tokens[right_pos], self.tokens[right_pos + 1])
            if pair in self.merge_ranks:
                rank = self.merge_ranks[pair]
                heapq.heappush(self.heap, (rank, right_pos))
                self.valid_positions.add(right_pos)
                print(f'  Added: position={right_pos}, pair={pair}, rank={rank}')

    def merge_step(self) -> bool:
        # 1. 获取最佳合并对
        result = self.get_best_pair()
        if result is None:
            return False
        
        position, pair, rank = result
        print(f'\nMerging: position={position}, pair={pair}, rank={rank}')

        # 2. 执行合并
        merged_token = pair[0] + pair[1]
        self.tokens[position] = merged_token
        del self.tokens[position + 1]
        print(f'  After merge: {self.tokens}')

        # 3. 使受影响的pair无效
        print(f'  Invalidating affected pairs')
        self._invalidate_affected_pairs(position)

        # 4. 添加新产生的pair
        print(f'  Adding new pairs')
        self._add_new_pairs(position)

        print(f'  Current valid positions: {self.valid_positions}')
        return True
    
    def merge_all(self) -> List[str]:
        """合并所有可能的pair"""
        step = 0
        while self.merge_step():
            step += 1
            print(f"Finished step {step} merging\n{'='*50}")

        return self.tokens
    
if __name__ == '__main__':
    merge_ranks = {
        ('n', 'e'): 0,
        ('ne', 'w'): 1,
        ('r', 'e'): 2,
        ('s', 'e'): 3,
        ('se', 't'): 4,
        ('re', 'new'): 5,
        ('re', 'set'): 6
    }

    tokens = ['r', 'e', 'n', 'e', 'w']

    print(f"初始tokens: {tokens}")
    print(f"合并优先级: {merge_ranks}")
    print("=" * 60)
    
    tokenizer = EfficientTokenizer(tokens, merge_ranks)
    result = tokenizer.merge_all()

    print(f"\n合并结果: {result}")

    