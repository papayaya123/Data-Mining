# map_plotly.py（可放進你的 notebook 或 cluster.py 旁）
import pandas as pd
import plotly.express as px
import us  # pip install us
from scipy.cluster.hierarchy import fcluster
from typing import List
import numpy as np

def plot_us_clusters_plotly(Z: np.ndarray, labels: List[str], k: int = 7, outfile: str = "us_clusters.html"):
    """用 Plotly 畫互動式美國 Choropleth；labels 必須是州全名（和 votes.repub 相同）。"""
    # SciPy 切群
    clus = fcluster(Z, t=k, criterion='maxclust')
    s = pd.Series(clus, index=labels, name="cluster").sort_index()

    # 州名→縮寫
    name_to_abbr = us.states.mapping('name', 'abbr')
    df_map = s.reset_index().rename(columns={'index': 'state'})
    df_map['abbr'] = df_map['state'].map(name_to_abbr)

    # 檢查缺失（例如名稱不匹配）
    missing = df_map[df_map['abbr'].isna()]
    if not missing.empty:
        raise ValueError(f"無法對應州縮寫：{missing['state'].tolist()}")

    # 確保離散色盤
    df_map['cluster'] = df_map['cluster'].astype(str)

    fig = px.choropleth(
        df_map,
        locations="abbr",
        color="cluster",
        locationmode="USA-states",
        scope="usa",
        hover_name="state",
        title=f"US Clusters (k={k})",
        color_discrete_sequence=px.colors.qualitative.Set3  # 可省略；讓顏色好看一點
    )
    fig.update_layout(margin=dict(l=0,r=0,t=40,b=0))
    fig.write_html(outfile)
    print(f"✅ 已輸出互動地圖：{outfile}")
    return df_map
