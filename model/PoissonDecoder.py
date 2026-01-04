import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Poisson


class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x)-1, min=1e-5, max=1e6)
    

class PoissonDecoder(nn.Module):
    def __init__(
            self, 
            feats_dim:int,
            ):
        super().__init__()
        self.dec_mean = nn.Sequential(nn.Linear(feats_dim, 1), nn.Sigmoid())
        self.dec_mean_act = MeanAct()

    def forward(
        self,
        graph,
        c_feat,
        r_feat,
        ckey='cell',
        rkey='region'
        ):
        """
        Parameters
        ----------

        """
        with graph.local_scope():
            graph.nodes[ckey].data['h'], graph.nodes[rkey].data['h'] = c_feat, r_feat
            graph.nodes[ckey].data['one'] = torch.ones([c_feat.shape[0], 1], device=c_feat.device)
            graph.nodes[rkey].data['one'] = torch.ones([r_feat.shape[0], 1], device=r_feat.device)

            # exp_graph = graph['cell', 'access', 'region'] if self.gene_similarity else graph

            graph.apply_edges(fn.u_mul_v('h', 'h', 'h_d'))
            graph.apply_edges(fn.u_mul_v('one', 'rs_factor', 'rs_factor'))
            graph.apply_edges(fn.u_mul_v('cs_factor', 'one', 'cs_factor'))

            h_d = graph.edata['h_d']
            mu_ = self.dec_mean(h_d)
            mu_ = graph.edata['rs_factor'] * mu_
            mu  = graph.edata['cs_factor'] * self.dec_mean_act(mu_)
        return mu

    def forward1(
        self,
        graph,
        c_feat,
        r_feat,
        rs_factor,
        ckey='cell',
        rkey='region',
    ):
        with graph.local_scope():
            graph.nodes[ckey].data['h'], graph.nodes[rkey].data['h'] = c_feat, r_feat
            graph.nodes[ckey].data['one'] = torch.ones([c_feat.shape[0], 1], device=c_feat.device)
            graph.nodes[rkey].data['one'] = torch.ones([r_feat.shape[0], 1], device=r_feat.device)

            graph.nodes[rkey].data['rs_factor1'] = rs_factor

            graph.apply_edges(fn.u_mul_v('h', 'h', 'h_d'))
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            graph.apply_edges(fn.u_mul_v('one', 'rs_factor1', 'rs_factor1'))
            graph.apply_edges(fn.u_mul_v('cs_factor', 'one', 'cs_factor'))

            h_d = graph.edata['h_d']
            mu_ = self.dec_mean(h_d)
            mu_ = graph.edata['rs_factor1'] * mu_
            mu  = graph.edata['cs_factor'] * self.dec_mean_act(mu_)
            score = graph.edata['score']

        return mu, score
    
    def loss(self, pred, truth=None):
        if truth is None:
            return -Poisson(pred[0]).log_prob(torch.zeros_like(pred[0])).mean()
        else:
            return -Poisson(pred[0]).log_prob(truth).mean()
