import dgl
import torch
from dgl import function as fn
from torch import nn
from dgl.nn.pytorch.conv import GATConv


class LightGraphConv(nn.Module):
    def __init__(self, drop_out=0.1, cell2cell=False):
        super().__init__()
        self.dropout = nn.Dropout(drop_out)
        self.cell2cell = cell2cell

    def forward(self, graph, feats):
        # feats: Tensor (同类型图) 或 (src, dst) 元组（二部图）
        if isinstance(feats, tuple):
            src_feats, _ = feats
        else:
            src_feats = feats

        with graph.local_scope():
            # 选择度归一化键名；优先使用 cii/cjj，不存在时回退到 ci/cj
            if self.cell2cell:
                cj_key, ci_key = 'cjj', 'cii'
            else:
                cj_key, ci_key = 'cj', 'ci'

            try:
                cj = graph.srcdata[cj_key]
                ci = graph.dstdata[ci_key]
            except KeyError:
                # 回退（以防没有分别为 cell-cell 单独存的度）
                cj = graph.srcdata['cj']
                ci = graph.dstdata['ci']

            # 保证设备/类型一致
            cj = cj.to(src_feats.device, dtype=src_feats.dtype)
            ci = ci.to(src_feats.device, dtype=src_feats.dtype)

            # 源特征乘以（丢弃后的）cj
            cj_dropout = self.dropout(cj)
            weighted_feats = src_feats * cj_dropout
            graph.srcdata['h'] = weighted_feats

            # 简单消息聚合，再乘以 ci
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'out'))
            out = graph.dstdata['out'] * ci
            return out


class lightgraphconvlayer(nn.Module):
    def __init__(
            self, 
            drop_out=0.1, 
            # alpha=None
            use_cell2cell = False
            ):
        super().__init__()
        """lightgraphconv layer

        drop_out : float
            dropout rate (feature dropout)
        alpha: float
            weight for gene massage
        """
        # self.alpha = alpha
        self.use_cell2cell = use_cell2cell
        conv = {}

        cell_to_region_key = 'access'
        region_to_cell_key = 'reverse-access'
        # gene_to_gene_key = 'co-exp'
        cell_to_cell_key = 'spatial-neighbor'

        # convolution on cell -> gene graph
        conv[cell_to_region_key] = LightGraphConv(drop_out=drop_out)

        # convolution on gene -> cell graph
        conv[region_to_cell_key] = LightGraphConv(drop_out=drop_out)

        #  convolution on cell -> cell graph
        if self.use_cell2cell:
            conv[cell_to_cell_key] = LightGraphConv(drop_out=drop_out, cell2cell=True)


        # convolution on gene -> gene graph
        # if self.alpha is not None:
        #     conv[gene_to_gene_key] = LightGraphConv(drop_out=drop_out, gene2gene=True)

        self.conv = dgl.nn.HeteroGraphConv(conv, aggregate='stack')
        # self.feature_dropout = nn.Dropout(drop_out)

    def forward(self, graph, c_feat, r_feat, ckey='cell', rkey='region'):
        """
        Paramters
        ---------
        graph : dgl.graph
        c_feat, r_feat : torch.FloatTensor
            node features
        ckey, grkey : str
            target node types

        Returns
        -------
        c_feat, r_feat : torch.FloatTensor
            output features

        Notes
        -----
        1. message passing
            MP_{i} = \{ MP_{i, r_{1}}, MP_{i, r_{2}}, ... \}
        2. aggregation
            h_{i} = \sigma_{j \in N(i) , r} MP_{i, j, r}
        """
        feats = {
            ckey: c_feat,
            rkey: r_feat
        }

        out = self.conv(graph, feats)
        c_feat = out[ckey].squeeze()
        r_feat = out[rkey].squeeze()

        return c_feat, r_feat
    

class GAT(nn.Module):
    def __init__(self, 
                 in_feats: int,
                 n_heads: int,):
        super().__init__()

        conv = {}

        cell_to_region_key = 'access'
        region_to_cell_key = 'reverse-access'
        # gene_to_gene_key = 'co-exp'

        # convolution on cell -> gene graph
        conv[cell_to_region_key] = GATConv(in_feats, in_feats, n_heads)

        # convolution on gene -> cell graph
        conv[region_to_cell_key] = GATConv(in_feats, in_feats, n_heads)


        # convolution on gene -> gene graph
        # if self.alpha is not None:
        #     conv[gene_to_gene_key] = LightGraphConv(drop_out=drop_out, gene2gene=True)

        self.conv = dgl.nn.HeteroGraphConv(conv, aggregate='stack')


    def forward(self, graph, c_feat, r_feat, ckey='cell', rkey='region'):

        feats = {
            ckey: c_feat,
            rkey: r_feat
        }

        out = self.conv(graph, feats)
        c_feat = out[ckey].squeeze()
        r_feat = out[rkey].squeeze()

        return c_feat.mean(1), r_feat.mean(1)



