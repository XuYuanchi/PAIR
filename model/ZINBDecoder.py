import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x) - 1., min=1e-5, max=1e6)


class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)

class ZINBDecoder(nn.Module):
    def __init__(self, feats_dim):
        super().__init__()
        """ZINB decoder for link prediction
        predict link existence (not edge type)
        """
        self.dec_mean = nn.Sequential(nn.Linear(feats_dim, 1), nn.Sigmoid())
        self.dec_disp = nn.Linear(feats_dim, 1)
        self.dec_disp_act = DispAct()
        self.dec_pi = nn.Sequential(nn.Linear(feats_dim, 1), nn.Sigmoid())
        self.dec_mean_act = MeanAct()

    def forward(self, graph, c_feat, g_feat, ckey='cell', gkey='region'):
        """
        Paramters
        ---------
        graph : dgl.homograph
        c_feat : torch.FloatTensor
            cell features
        g_feat : torch.FloatTensor
            gene features
        g_last : torch.FloatTensor
            gene features of the last laye
        ckey, gkey : str
            target node types

        Returns
        -------
        mu : torch.FloatTensor
            the estimated mean of ZINB model shape : (n_edges, 1)
        disp : torch.FloatTensor
            the estimated dispersion of ZINB model shape : (n_edges, 1)
        pi : torch.FloatTensor
            the estimated dropout rate of ZINB model shape : (n_edges, 1)
        ge_score : torch.FloatTensor
            the predicted values of highly correlated gene edges when considering gene massage
        """
        ge_score = None

        with graph.local_scope():
            graph.nodes[ckey].data['h'], graph.nodes[gkey].data['h'] = c_feat, g_feat
            graph.nodes[ckey].data['one'] = torch.ones([c_feat.shape[0], 1], device=c_feat.device)
            graph.nodes[gkey].data['one'] = torch.ones([g_feat.shape[0], 1], device=g_feat.device)

            graph.apply_edges(fn.u_mul_v('h', 'h', 'h_d'))
            graph.apply_edges(fn.u_mul_v('one', 'rs_factor', 'rs_factor'))
            graph.apply_edges(fn.u_mul_v('cs_factor', 'one', 'cs_factor'))

            h_d = graph.edata['h_d']
            mu_ = self.dec_mean(h_d)
            disp_ = self.dec_disp(h_d)
            pi = self.dec_pi(h_d)

            disp = self.dec_disp_act(graph.edata['rs_factor'] * disp_)
            mu_ = graph.edata['rs_factor'] * mu_
            mu = graph.edata['cs_factor'] * self.dec_mean_act(mu_)

        return mu, disp, pi