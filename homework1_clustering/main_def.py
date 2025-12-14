import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

def plot_hierarchical_clustering(data: np.ndarray, 
                                 cluster_nums=[2,3,4,5], 
                                 linkage_method='ward'):
    """
    層次式分群視覺化工具
    -----------------------
    參數：
        data : np.ndarray
            二維資料，例如 shape = (n_samples, 2)
        cluster_nums : list[int]
            要比較的分群數（預設為 [2,3,4,5]）
        linkage_method : str
            linkage 方法 ('ward', 'complete', 'average', 'single')

    輸出：
        1. Dendrogram（樹狀圖）
        2. 原始資料散點圖
        3. 各群數 k 的分群結果子圖
    """
    
    # 🧩 檢查資料維度
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("⚠️ data 必須是 shape=(n_samples, 2) 的 numpy array")

    # 🈶 設定中文字體（macOS）
    plt.rcParams['font.family'] = 'Heiti TC'
    plt.rcParams['axes.unicode_minus'] = False

    # -------------------------------
    # 🔹 Step 1: Dendrogram 樹狀圖
    # -------------------------------
    linked = linkage(data, method=linkage_method)
    plt.figure(figsize=(6, 5))
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False)
    plt.title(f'階層式分群樹狀圖 (method={linkage_method})')
    plt.xlabel('樣本點')
    plt.ylabel('距離')
    plt.grid(True)
    plt.show()

    # -------------------------------
    # 🔹 Step 2: 原始資料散點圖
    # -------------------------------
    plt.figure(figsize=(6, 5))
    plt.scatter(data[:, 0], data[:, 1], s=50, edgecolors='k')
    plt.title('原始資料散點圖')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.show()

    # -------------------------------
    # 🔹 Step 3: 各群數子圖
    # -------------------------------
    n = len(cluster_nums)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.ravel()

    for i, num in enumerate(cluster_nums):
        hc = AgglomerativeClustering(n_clusters=num, linkage=linkage_method)
        cluster_label = hc.fit_predict(data)

        sc = axes[i].scatter(
            data[:, 0],
            data[:, 1],
            c=cluster_label,
            cmap='viridis',
            s=60,
            edgecolors='k'
        )
        axes[i].set_title(f'Hierarchical Clustering (k={num})')
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')
        axes[i].grid(True)

    # 共用 colorbar
    fig.colorbar(sc, ax=axes, orientation='vertical', fraction=0.02, pad=0.04, label='Cluster Label')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

def compare_kmeans_pam(data: np.ndarray, k_values=[2, 3, 4, 5]):
    """
    比較 K-Means 與 PAM (K-Medoids) 的分群結果
    
    參數：
    ---------
    data : np.ndarray
        shape=(n_samples, 2) 的資料。
    k_values : list[int]
        要測試的群數，例如 [2,3,4,5]
    
    功能：
    ---------
    1. 上排顯示 K-Means 分群結果
    2. 下排顯示 PAM (K-Medoids) 分群結果
    3. 每個群的中心以紅色 X 標示
    """
    
    # --- 🧩 檢查資料格式 ---
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("⚠️ data 必須是 shape=(n_samples, 2) 的 numpy array")
    
    # --- 🧠 設定中文字體（macOS 可用 Heiti TC）---
    plt.rcParams['font.family'] = 'Heiti TC'
    plt.rcParams['axes.unicode_minus'] = False

    # --- 🎨 建立子圖 ---
    fig, axes = plt.subplots(2, len(k_values), figsize=(4 * len(k_values), 8), constrained_layout=True)
    axes = axes.ravel()

    # --- 🔹 K-Means 上排 ---
    for i, k in enumerate(k_values):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        kmeans.fit(data)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        axes[i].scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=40, edgecolors='k')
        axes[i].scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='red')
        axes[i].set_title(f'K-Means (k={k})', fontsize=12)
        axes[i].set_xlabel('Feature 1'); axes[i].set_ylabel('Feature 2'); axes[i].grid(True)

    # --- 🔹 PAM (K-Medoids) 下排 ---
    for i, k in enumerate(k_values):
        kmedoids = KMedoids(n_clusters=k, random_state=0, metric='euclidean')
        kmedoids.fit(data)
        labels = kmedoids.labels_
        medoids = kmedoids.cluster_centers_

        axes[i + len(k_values)].scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=40, edgecolors='k')
        axes[i + len(k_values)].scatter(medoids[:, 0], medoids[:, 1], marker='X', s=200, c='red')
        axes[i + len(k_values)].set_title(f'PAM (K-Medoids) (k={k})', fontsize=12)
        axes[i + len(k_values)].set_xlabel('Feature 1'); axes[i + len(k_values)].set_ylabel('Feature 2'); axes[i + len(k_values)].grid(True)

    # --- 🧭 標註整體說明 ---
    fig.suptitle('K-Means vs PAM (K-Medoids) 分群比較', fontsize=16, y=1.02)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# -----------------------------
# 🔹 第一部分：執行 DBSCAN 並印出分群結果
# -----------------------------
def run_dbscan(data, eps=0.3, min_samples=10, scale=True):
    """
    執行 DBSCAN 分群並輸出基本統計
    參數:
        data : np.ndarray
            輸入資料 (n_samples, n_features)
        eps : float
            鄰近距離閾值
        min_samples : int
            最少鄰居數
        scale : bool
            是否進行標準化 (StandardScaler)
    回傳:
        db : 訓練好的 DBSCAN 模型
        labels : 分群標籤
    """
    # 資料標準化（可關閉）
    X = StandardScaler().fit_transform(data) if scale else data

    # 執行 DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    # 統計資訊
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print(f"Estimated number of clusters: {n_clusters_}")
    print(f"Estimated number of noise points: {n_noise_}")

    return db, labels, X


# -----------------------------
# 🔹 第二部分：畫出分群結果圖
# -----------------------------
def plot_dbscan_clusters(db, labels, X):
    """
    畫出 DBSCAN 的分群結果
    參數:
        db : 已訓練的 DBSCAN 模型
        labels : 模型分群結果
        X : (標準化後) 資料座標
    """
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    plt.figure(figsize=(7, 6))
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # 黑色代表雜訊點
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        # 核心點
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        # 非核心點
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"DBSCAN Clusters (n={len(set(labels)) - (1 if -1 in labels else 0)})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()
