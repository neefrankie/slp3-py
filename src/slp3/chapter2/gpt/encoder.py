# Adapted from https://github.com/openai/tiktoken/blob/main/src/lib.rs

from dataclasses import dataclass
from typing import Dict, List, NamedTuple
import heapq
import regex

class MergePart(NamedTuple):
    """
    The position of a byte in bytes and its rank.
    """
    position: int # points to the position of a byte
    rank: int | float # the rank for current position + next part.

class BytePairMerger:
    def __init__(self, piece: bytes, ranks: Dict[bytes, int]):
        self.piece = piece
        self.ranks = ranks
        self.parts, self.min_rank = self.build_initial_ranks()

    def get_pair_rank(self, i: int) -> int | float:
        pair = self.piece[i:i+2]
        return self.ranks.get(pair, float('inf'))
    
    def get_post_merge_rank(self, i: int):
        # When we merge two bytes, we should invalidate its
        # left and right ranks.
        # We should use the self.parts pointers to calculate how many
        # bytes to span.
        # At this point the merged position is not deleted yet,
        # so we should span 3 positions.
        if i + 3 < len(self.parts):
            # self.parts[i] and self.parts[i+1] should be assumed merged.
            # Therefore next pair is self.parts[i:i+2] and self.parts[i+2]
            start = self.parts[i].position
            end = self.parts[i+3].position
            pair = self.piece[start:end]
            return self.ranks.get(
                pair, 
                float('inf')
            )
        
        return float('inf')

    def build_initial_ranks(self):
        # Initially we assign a rank to each consecutive byte pair.
        parts: List[MergePart] = []
        min_rank = MergePart(-1, float('inf'))

        # For hello, it would be:
        # bytes:    h  e  l  l  o
        # position: 0  1  2  3  4
        # ranks:    20 13 24 30 inf inf
        for i in range(len(self.piece) - 1):
            rank = self.get_pair_rank(i)
            if rank < min_rank.rank:
                min_rank = MergePart(i, rank)
            parts.append(MergePart(i, rank))
        
        parts.append(MergePart(len(self.piece) - 1, float('inf')))
        # parts has one more element than piece so that
        # the final byte could be included when performming i+3.
        # It aslo acts as a sentinel when slicing the original bytes finally.
        parts.append(MergePart(len(self.piece), float('inf')))
        return parts, min_rank
    
    def apply_merge(self):
        i = self.min_rank.position
        if i > 0:
            self.parts[i-1] = MergePart(
                position=i-1,
                rank=self.get_post_merge_rank(i-1)
            )

        self.parts[i] = MergePart(
            position=i,
            rank=self.get_post_merge_rank(i)
        )
        del self.parts[i+1]

    def update_min_rank(self):
        min_rank = MergePart(-1, float('inf'))
        for j in range(len(self.parts) - 1):
            if self.parts[j].rank < min_rank.rank:
                min_rank = MergePart(j, self.parts[j].rank)

        self.min_rank = min_rank

    def merge_until_done(self):
        while self.min_rank.rank != float('inf'):
            self.apply_merge()
            self.update_min_rank()

        return self

    def iter_parts(self):
        # Slice the original bytes delimited by MergePart.
        for start, end in zip(self.parts[:-1], self.parts[1:]):
            yield self.piece[start.position:end.position]

    def encode(self) -> List[int]:
        # Slice the original bytes delimited by MergePart.position.
        # Note that the MergPart.rank after performing merge is not valid,
        # which are adjacent pair's rank, not the rank of each token.
        return [self.ranks[part] for part in self.iter_parts()]
    
# Adapted from struct State of tiktoken lib.rs.
# The original data structure:
# struct State {
#     prev: usize,
#     end: usize,
#     next_end: usize,
#     next_rank: Rank,
#     cur_rank: Rank,
# }
# makes me headache with so much pointers,
# so I simplified it by removing next_end.
@dataclass
class Node:
    prev_idx: int # Index of previous node
    end_idx: int  # Next valid node. It's also the end position of the byte slice represnted by current node.
    cur_rank: int | float = float('inf') # Rank of current node if it's merged.
    next_rank: int | float = float('inf') # Merging rank of current node with next valid node.

class Merge(NamedTuple):
    rank: int
    start: int
    

class BytePairMergeLarge:
    def __init__(self, piece: bytes, ranks: Dict[bytes, int]) -> None:
        self.piece = piece
        self.ranks = ranks
        self.states, self.heap = self.init_ranks()
        pass

    def get_pair_rank(self, i: int) -> int | None:
        pair = self.piece[i:i+2]
        return self.ranks.get(pair, None)

    def init_ranks(self):
        states = [
            Node(
                prev_idx=-1,
                end_idx=1,
            )
        ]
        h: List[Merge] = []
        for i in range(len(self.piece) - 1):
            rank = self.get_pair_rank(i)
            if rank is not None:
                heapq.heappush(
                    h, 
                    Merge(rank=rank, start=i)
                )
                states[i].next_rank = rank

            states.append(
                Node(
                    prev_idx=i,
                    end_idx=i+2, # the final element will have end_idx == len(self.piece)
                )
            )

        return states, h
    
    def potential_merge(self, start_idx: int, upto_idx: int):
        """
        Check whether the token starting with start_idx 
        is mergeable with next token at upto_idx
        
        Args:
            start_idx: the index of the starting node
            upto_idx: the index of the node to be merged in next round.
        """
        self.states[start_idx].next_rank = float('inf')
        # Ensure the max upto_idx points to the last element.
        if upto_idx >= len(self.states):
            return
        # Slicing includes the node at upto_idx, so we need to find
        # the next node's starting index.
        end_idx = self.states[upto_idx].end_idx
        # If end_idx execeeds len(self.piece), no op.
        # This shouldn't happen, but just in case.
        if end_idx > len(self.piece):
            return
        # Get the slice from node at start_indx until node at upto_idx
        pair = self.piece[start_idx:end_idx]
        # Try to find the rand of this slice.
        rank = self.ranks.get(pair, None)
        # If the rank exiss, it means they could be merged in future rounds.
        if rank is not None:
            heapq.heappush(
                self.heap, 
                Merge(rank=rank, start=start_idx)
            )
            # Update future merge rank.
            self.states[start_idx].next_rank = rank
    
    def apply_merge(self, left: Merge):
        # left_start_idx will be merged with right_start_idx
        left_start_idx = left.start
        right_start_idx = self.states[left_start_idx].end_idx
        # The merged tokens should form a new relationship with right_end_idx.
        # When right_start_idx is the last element, right_end_idx will be len(self.piece)
        right_end_idx = self.states[right_start_idx].end_idx

        self.states[left_start_idx].cur_rank = self.states[left_start_idx].next_rank
        self.states[left_start_idx].end_idx = right_end_idx
        self.potential_merge(left_start_idx, right_end_idx)
        
        if right_end_idx < len(self.piece):
            self.states[right_end_idx].prev_idx = left_start_idx
        
        if left_start_idx > 0:
            prev_start_idx = self.states[left_start_idx].prev_idx
            self.potential_merge(prev_start_idx, left_start_idx)

        # Invalidate the merge right_start_idx
        self.states[right_start_idx].next_rank = float('inf')

    def merge_until_done(self):
        while self.heap:
            left = heapq.heappop(self.heap)
            if left.rank == float('inf'):
                break
            if left.rank != self.states[left.start].next_rank:
                continue

            self.apply_merge(left)

        return self

    def encode(self):
        result: List[int] = []
        i = 0
        while i < len(self.states):
            if self.states[i].cur_rank != float('inf'):
                result.append(int(self.states[i].cur_rank))
            else:
                b = self.piece[i:self.states[i].end_idx]
                result.append(self.ranks[b])
            i = self.states[i].end_idx

        return result
    
def byte_pair_encode(piece: bytes, ranks: Dict[bytes, int]) -> List[int]:
    piece_len = len(piece)
    if piece_len == 1:
        return [ranks[piece]]
    
    if len(piece) < 100:
        return BytePairMerger(piece, ranks).merge_until_done().encode()
    

    return BytePairMergeLarge(piece, ranks).merge_until_done().encode()

class Encoding:
    def __init__(self, encoder: Dict[bytes, int], pattern: str):
        self.encoder = encoder
        self.decoder: Dict[int, bytes] = {
            v: k 
            for k, v in encoder.items()
        }
        self.regex = regex.compile(pattern)

    def decode_bytes(self, tokens: List[int]) -> List[bytes]:
        ret: List[bytes] = []
        for token in tokens:
            if token not in self.decoder:
                ret.append(b'<unk>')
            else:
                ret.append(self.decoder[token])

        return ret
    
    def decode_readable(self, ids: List[int]) -> List[str]:
        readable: List[str] = []
        byte_chunks = self.decode_bytes(ids)
        for b in byte_chunks:
            try:
                s = b.decode('utf-8')
                if not s.isprintable():
                    s = repr(s)[1:-1]
                readable.append(s)
            except UnicodeDecodeError:
                readable.append(b.hex())

        return readable
    
    def decode(self, tokens: List[int]) -> str:
        return b"".join(self.decode_bytes(tokens)).decode('utf-8', errors="replace")
    
    # Adapted from lib.rs CoreBPE.encode_ordinary
    def encode_ordinary(self, text: str) -> List[int]:
        ret: List[int] = []
        for chunk in self.regex.findall(text):
            piece: bytes = chunk.encode('utf-8')
            if rank := self.encoder.get(piece, None):
                ret.append(rank)
            else:
                ret.extend(byte_pair_encode(piece, self.encoder))
        return ret
    
    # Adapted from core.py Encoding.encode
    def encode(self, text: str) -> List[int]:
        text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
        return self.encode_ordinary(text)