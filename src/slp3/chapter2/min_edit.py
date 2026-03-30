from enum import Enum
from typing import List, NamedTuple


class EditAction(Enum):
    DEL = 1
    INS = 2
    SUB = 3
    COPY = 4

class BackPointer(NamedTuple):
    prev_row: int
    prev_col: int
    action: EditAction
    char_info: str | None = None

def min_edit_distance(source: str, target: str):
    # n is row, m is column.
    n, m = len(source), len(target)

    # 1. 代价矩阵：只负责存数字，初始化全为0
    # 这样 dp[i-1][j] + 1 这种操作非常安全，不需要判空
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # 2. 路径矩阵：负责存结构体
    # 初始化为 None，因为 (0,0) 没有前驱
    parent: List[List[BackPointer | None]] = [
        [None] * (m + 1) for _ in range(n + 1)
    ]

    # --- 初始化 ---
    for i in range(1, n+1):
        dp[i][0] = dp[i-1][0] + 1
        parent[i][0] = BackPointer(
            prev_row=i-1, 
            prev_col=0, 
            action=EditAction.DEL,
            char_info=source[i-1]
        )

    # Source is empty, the cost of inserting every target character.
    for j in range(1, m+1):
        dp[0][j] = dp[0][j-1] + 1
        parent[0][j] = BackPointer(
            prev_row=0, 
            prev_col=j-1, 
            action=EditAction.INS,
            char_info=target[j-1]
        )

    # --- 填表 ---
    for i in range(1, n+1):
        for j in range(1, m+1):
            # 计算三种可能的代价
            cost_del = dp[i-1][j] + 1
            cost_ins = dp[i][j-1]+1
            cost_sub_or_copy = dp[i-1][j-1] + (0 if source[i-1] == target[j-1] else 2)

            # 找出最小值
            min_cost = min(cost_del, cost_ins, cost_sub_or_copy)
            
            dp[i][j] = min_cost

            # 根据最小值来源，更新路径矩阵
            if min_cost == cost_sub_or_copy:
                action = EditAction.COPY if source[i-1] == target[j-1] else EditAction.SUB
                parent[i][j] = BackPointer(
                    prev_row=i-1, 
                    prev_col=j-1, 
                    action=action,
                    char_info=f"'{source[i-1]}' -> '{target[j-1]}'"
                )
            elif min_cost == cost_del:
                parent[i][j] = BackPointer(
                    prev_row=i-1, 
                    prev_col=j, 
                    action=EditAction.DEL,
                    char_info=source[i-1]
                )
            else:
                parent[i][j] = BackPointer(
                    prev_row=i, 
                    prev_col=j-1, 
                    action=EditAction.INS,
                    char_info=target[j-1]
                )

    return dp, parent

def backtrace(parent: List[List[BackPointer | None]], source: str, target: str):
    curr_i, curr_j = len(source), len(target)
    steps: List[str] = []

    while parent[curr_i][curr_j] is not None:
        bp = parent[curr_i][curr_j]
        assert bp is not None
        steps.append(f"{bp.action.name}: {bp.char_info}")
        curr_i, curr_j = bp.prev_row, bp.prev_col

    steps.reverse()

    return steps


def print_matrix(dp: List[List[int]], source: str, target: str):
    # 1. 定义列宽，保证对齐
    header_col_width = 7
    col_width = 2
    
    # 2. 构建表头
    # 左上角的空白占位 + 列名
    header = "Src\\Tar".center(header_col_width) + " | "

    # 加上 target 的字符列
    header += "#".center(col_width) + " | " # 第一列是空串的情况
    for char in target:
        header += f"{char}".center(col_width) + " | "

    # 打印分割线
    print(header)
    print("-" * len(header))

    # 3. 构建数据行
    for i, row in enumerate(dp):
        row_str = ""

        # 行首：显示 source 的字符
        if i == 0:
            row_str += "#".center(header_col_width) + " | " # 第一行是空串的情况
        else:
            row_str += f"{source[i-1]}".center(header_col_width) + " | "

        # 数据单元格
        for val in row:
            row_str += f"{val}".center(col_width) + " | "

        print(row_str)


if __name__ == "__main__":
    source = "intention"
    target = "execution"

    dp, parent = min_edit_distance(source, target)

    print_matrix(dp, source, target)

    steps = backtrace(parent, source, target)
    for step in steps:
        print(step)