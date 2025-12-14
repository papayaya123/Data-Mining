import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.cluster.hierarchy import dendrogram , linkage


def distance_matrix(points):
    n = len(points)
    D = [[0.0]*n for _ in range(n)]
    for i, (xi, yi) in enumerate(points):
        for j, (xj, yj) in enumerate(points):
            dx = xi - xj
            dy = yi - yj
            D[i][j] = math.hypot(dx, dy)  # sqrt(dx*dx + dy*dy)
    return D

def nearest_pair_from_D_np_alt(D):
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 points")
    iu, ju = np.triu_indices(n, k=1)  # 直接拿上三角座標
    vals = D[iu, ju]                  # 抽出上三角距離的一維陣列
    k = np.argmin(vals)
    return int(iu[k]), int(ju[k]), float(vals[k])


