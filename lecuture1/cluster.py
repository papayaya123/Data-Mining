# cluster.py
from __future__ import annotations
from typing import Dict, Tuple, List, Iterable
import math
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.cluster.hierarchy import dendrogram, fcluster
import matplotlib.pyplot as plt


# ========= 你原本的核心步驟（保留語意） =========
def find_cluster(df: pd.DataFrame) -> Dict[str, Tuple[str, float]]:
    """對每個點找最近鄰（排除自己）。回傳 {節點: (最近鄰, 距離)}"""
    find_friend: Dict[str, Tuple[str, float]] = {}
    for idx, row in df.iterrows():
        row_no_self = row.drop(labels=[idx])        # 只排除自己
        friend = row_no_self.idxmin()
        distance = float(row_no_self.min())
        find_friend[idx] = (friend, distance)
    return find_friend


def find_min_value_in_cluster(find_friend: dict) -> float:
    """取全域最小距離"""
    min_val = float('inf')
    for _, (_, d) in find_friend.items():
        if d < min_val:
            min_val = d
    return min_val


def groups_from_nearest(
    friend: Dict[str, Tuple[str, float]],
    min_val: float,
    tol: float = 1e-9
) -> List[Tuple[str, ...]]:
    """
    取出本輪要合併的群：
      - 互為最近鄰且距離≈min_val 的配對 → 以 2-tuple 表示
      - 其餘點 → 以 1-tuple 表示
      - 依 friend.keys() 的原順序排序
    """
    # 1) 先過濾距離≈min_val 的邊
    filtered = {a: (b, d) for a, (b, d) in friend.items()
                if math.isclose(d, min_val, abs_tol=tol)}

    # 2) 只保留「互為最近鄰」的無向邊，並去重
    edges: set[Tuple[str, str]] = set()
    for a, (b, _) in filtered.items():
        bb = filtered.get(b)
        if bb and bb[0] == a:
            edges.add(tuple(sorted((a, b))))  # ('B','C') 與 ('C','B') 合併

    # 3) 為了輸出穩定，依原鍵順序排序
    order = {name: i for i, name in enumerate(friend.keys())}
    pairs = sorted(edges, key=lambda t: (order[t[0]], order[t[1]]))

    # 4) 把單點補進來（統一用 1-tuple）
    paired_nodes = {x for u, v in pairs for x in (u, v)}
    singles = [(n,) for n in friend.keys() if n not in paired_nodes]

    return pairs + singles


def centroid_linkage(df: pd.DataFrame, groups: Iterable[Tuple[str, ...]]) -> pd.DataFrame:
    """
    以「群內所有**數值欄**」取平均當作質心。支援任意維度資料。
    群標籤：單點用自身；多點用排序後 'A+B+...'
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cluster_dict: Dict[str, np.ndarray] = {}

    for g in groups:
        members = list(g) if isinstance(g, (tuple, list, set)) else [g]
        sub = df.loc[members, numeric_cols]
        centroid = sub.mean(axis=0).to_numpy(dtype=float)
        label = members[0] if len(members) == 1 else "+".join(sorted(members))
        cluster_dict[label] = centroid

    return pd.DataFrame.from_dict(cluster_dict, orient='index', columns=list(numeric_cols))


def fit_custom_hclust(
    df: pd.DataFrame,
    tol: float = 1e-9,
    max_iters: int = 100,
) -> tuple[pd.DataFrame, List[List[Tuple[str, ...]]], List[float]]:
    """
    主流程：反覆
      1) 距離矩陣（歐式）
      2) 對每點找最近鄰
      3) 取全域最小距離 → 形成「互為最近鄰 & 距離≈最小值」的 pair + singles
      4) 以群質心取代，進入下一輪
    回傳：final_df, levels(每輪群組), min_values(每輪高度)
    """
    DF = df.copy()
    levels: List[List[Tuple[str, ...]]] = []
    min_values: List[float] = []

    for _ in range(max_iters):
        if len(DF) <= 1:
            if len(DF) == 1:
                levels.append([(DF.index[0],)])
            break

        dist_mat = distance_matrix(DF.values, DF.values)
        dist_df = pd.DataFrame(dist_mat, columns=DF.index, index=DF.index)

        friend = find_cluster(dist_df)
        min_val = find_min_value_in_cluster(friend)

        groups = groups_from_nearest(friend, min_val, tol=tol)
        levels.append(groups)
        min_values.append(min_val)

        if all(len(g) == 1 for g in groups):
            break

        DF = centroid_linkage(DF, groups)

    return DF, levels, min_values


# ========= 把 levels/mins 轉 SciPy linkage，並提供畫樹/切樹 =========
def _ensure_non_decreasing(xs: List[float]) -> List[float]:
    """避免質心造成高度回降：強制非遞減（可選）。"""
    out, cur = [], -float('inf')
    for v in xs:
        cur = max(cur, v)
        out.append(cur)
    return out


def levels_to_linkage(
    levels: List[List[Tuple[str, ...]]],
    mins: List[float],
    init_labels: List[str],
    enforce_monotone: bool = True,
) -> np.ndarray:
    """
    把每一輪的並行合併轉成 SciPy linkage 矩陣 Z。
    同一輪的所有 pair 都使用該輪的 min 當高度。
    """
    if enforce_monotone:
        mins = _ensure_non_decreasing(mins)

    # 葉節點 id 與 size=1
    label_to_id: Dict[str, int] = {lab: i for i, lab in enumerate(init_labels)}
    counts: Dict[int, int] = {i: 1 for i in range(len(init_labels))}
    next_id = len(init_labels)

    rows: List[List[float]] = []
    for round_groups, h in zip(levels, mins):
        for g in round_groups:
            if len(g) != 2:
                continue
            a, b = g
            if a not in label_to_id or b not in label_to_id:
                raise ValueError(f"Unknown label in levels: {g}")
            ia, ib = label_to_id[a], label_to_id[b]
            left, right = sorted((ia, ib))
            new_size = counts[left] + counts[right]
            rows.append([left, right, float(h), float(new_size)])

            # 新群 id 與標籤（沿用你的命名規則）
            new_id = next_id
            next_id += 1
            counts[new_id] = new_size
            new_label = "+".join(sorted([a, b]))
            label_to_id[new_label] = new_id
            label_to_id[f"({a}+{b})"] = new_id  # 可選別名

    return np.array(rows, dtype=float)


def plot_dendrogram(
    Z: np.ndarray,
    labels: List[str],
    figsize=(8, 4),
    **kwargs
):
    """用 linkage Z 畫 dendrogram。"""
    plt.figure(figsize=figsize)
    dendrogram(Z, labels=labels, **kwargs)
    plt.title("Dendrogram (custom parallel nearest merges)")
    plt.tight_layout()
    plt.show()


def cut_tree_k(Z: np.ndarray, labels: List[str], k: int) -> pd.Series:
    """依群數 k 切樹，回傳 Series(index=labels, values=cluster_id)。"""
    clus = fcluster(Z, t=k, criterion='maxclust')
    return pd.Series(clus, index=labels, name=f"k={k}")


__all__ = [
    # 你的核心
    "find_cluster", "find_min_value_in_cluster", "groups_from_nearest",
    "centroid_linkage", "fit_custom_hclust",
    # 轉接 & 視覺
    "levels_to_linkage", "plot_dendrogram", "cut_tree_k",
]
