import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import cKDTree


def construct_region_graph(gex_features, corr_method='cosine', corr_threshold=0.9):
    """Generate nodes, edges and edge weights for dataset.

    Parameters
    ----------
    gex_features: anndata.AnnData
        Gene data, contains feature matrix (.X) and feature names (.var['feature_types']).

    Returns
    --------
    uu: list[int]
        Predecessor node id of each edge.
    vv: list[int]
        Successor node id of each edge.
    ee: list[float]
        Edge weight of each edge.
    """

    if corr_method == 'pearson':
        corr = np.abs(np.corrcoef(gex_features, rowvar=False))
    elif corr_method == 'cosine':
        corr = cosine_similarity(gex_features.T)

    row, col = np.diag_indices_from(corr)
    corr[row, col] = 0

    coexp_edges = np.where(abs(corr) > corr_threshold)
    uncoexp_edges = np.where(abs(corr) < 1 - corr_threshold)
    # neg_idx = np.random.choice(len(nuu), 10*len(uu))
    # nuu, nvv = nuu[neg_idx], nvv[neg_idx]

    return coexp_edges, uncoexp_edges


def add_degree(graph, edge_types):
    def _calc_norm(x):
        x = x.numpy().astype('float32')
        x[x == 0.] = np.inf
        x = torch.FloatTensor(1. / np.sqrt(x))
        return x.unsqueeze(1)

    cell_ci, region_ci = _calc_norm(graph['reverse-access'].in_degrees()), _calc_norm(graph['access'].in_degrees())
    cell_cj, region_cj = _calc_norm(graph['access'].out_degrees()), _calc_norm(graph['reverse-access'].out_degrees())
    graph.nodes['cell'].data.update({'ci': cell_ci, 'cj': cell_cj})
    graph.nodes['region'].data.update({'ci': region_ci, 'cj': region_cj})

    if 'co-access' in edge_types:
        region_cii, region_cjj = _calc_norm(graph['co-access'].in_degrees()), _calc_norm(graph['co-access'].out_degrees())
        graph.nodes['region'].data.update({'cii': region_cii, 'cjj': region_cjj})

def build_spatial_cell_edges(
    adata,
    rad_cutoff=None,
    rad_coef=1.5,
    coor_key='spatial',
    include_self=False,
):
    """
    基于 adata.obsm[coor_key] 的半径邻接，返回 (src, dst)，双向边。
    半径：若 rad_cutoff 为 None，则用 rad_coef * 最小非零最近邻距离。
    """
    coords = np.asarray(adata.obsm[coor_key])
    if coords.ndim != 2 or coords.shape[0] == 0:
        raise ValueError(f"Invalid coordinates in adata.obsm['{coor_key}'].")

    tree = cKDTree(coords)
    if rad_cutoff is None:
        # 最近邻（排除自身）的最小距离
        nn_dists, _ = tree.query(coords, k=2)  # [:,0]==0(自身), [:,1]为最近他者
        min_nonzero = float(np.min(nn_dists[:, 1]))
        if not np.isfinite(min_nonzero) or min_nonzero <= 0:
            raise ValueError("无法自动估半径：没有非零点间距（可能所有点重合或仅1个点）。")
        radius = rad_coef * min_nonzero
    else:
        radius = float(rad_cutoff)

    # 以半径搜邻居（包含半径边界）
    neigh_lists = tree.query_ball_point(coords, r=radius)

    # 组装有向边（双向），并可选去掉自环
    src, dst = [], []
    for i, nbrs in enumerate(neigh_lists):
        for j in nbrs:
            if not include_self and i == j:
                continue
            src.append(i); dst.append(j)
            if i != j:     # 双向
                src.append(j); dst.append(i)

    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)

    # 简要统计
    deg = np.bincount(src, minlength=coords.shape[0])
    if include_self:
        print(f"[cell spatial graph] radius={radius:.4f}, avg deg (incl self): {deg.mean():.4f}")
    else:
        # 因为我们构造了双向边，度数≈邻居数*2（无自环时）；这里只打印度均值
        print(f"[cell spatial graph] radius={radius:.4f}, avg out-degree: {deg.mean():.4f}")

    return (src, dst), radius