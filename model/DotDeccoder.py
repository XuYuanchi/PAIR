import torch.nn as nn
import dgl.function as fn

class DotDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        """Dotproduct decoder for link prediction
        predict link existence (not edge type)
        """

    def forward1(self, graph, c_feat, g_feat, rs_factor, ckey='cell', gkey='region'):
        """
        Paramters
        ---------
        graph : dgl.homograph
        c_feat : torch.FloatTensor
            cell features
        g_feat : torch.FloatTensor
            gene features
        g_last : torch.FloatTensor
            gene features of the last layer
        ckey, gkey : str
            target node types

        Returns
        -------
        pred : torch.FloatTensor
            shape : (n_edges, 1)
        """

        with graph.local_scope():
            graph.nodes[ckey].data['h'] = c_feat
            graph.nodes[gkey].data['h'] = g_feat
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            pred = graph.edata['score']

        return pred
    
    def loss(self, pred, truth=None):
        return 0