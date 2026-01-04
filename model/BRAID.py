import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layer import lightgraphconvlayer
from .NBDecoder import NBDecoder
from .DotDeccoder import DotDecoder
from .PoissonDecoder import PoissonDecoder
from utils import NB_Loss
from utils.torch_clustering.kmeans.kmeans import PyTorchKMeans

class BRAID(nn.Module):
    def __init__(
        self,
        n_layers,
        n_cells,
        n_regions,
        drop_out,
        feats_dim,
        learnable_weight=False,
        ):
        super().__init__()
        """a novel scATAC-seq representation learning method based on graph node embedding
        ---------------------------------------------------------------------------------------
        n_layers : int
            number of GCN layers
        n_cells : int
            number of cells
        n_regions : int
            number of regions   
        drop_out : float
            dropout rate (neighbors)
        feats_dim : int
            node feature dimension
        learnable_weight : boolean
            whether to learn weights for embedding aggregation, if False, use 1/n_layers
        """
        self.n_cells = n_cells
        self.n_regions = n_regions

        self.cell_feature = nn.Parameter(torch.Tensor(self.n_cells, feats_dim))
        self.region_feature = nn.Parameter(torch.Tensor(self.n_regions, feats_dim))

        nn.init.xavier_uniform_(self.cell_feature)
        nn.init.xavier_uniform_(self.region_feature)

        self.region_factors = torch.nn.Parameter(torch.ones(self.n_regions))

        self.n_layers = n_layers
        self.encoders = nn.ModuleList()
        for _ in range(n_layers):
            self.encoders.append(lightgraphconvlayer(drop_out=drop_out))
            # self.encoders.append(GAT(in_feats=feats_dim, n_heads=2))

        if self.n_layers == 2:
            self.weights = torch.tensor([1., 1. / 2, 1. / 2])
        else:
            self.weights = torch.ones([self.n_layers + 1, 1]) / (self.n_layers + 1)

        if learnable_weight:
            self.weights = nn.Parameter(self.weights)

        # self.decoder = PoissonDecoder(feats_dim)
        self.decoder = NBDecoder(feats_dim)
        # self.decoder = ZINBDecoder(feats_dim)
        self.DotDecoder = DotDecoder()

        self.device = None

        for p, q in self.decoder.named_parameters():
            if 'weight' in p:
                nn.init.kaiming_normal_(q)
            elif 'bias' in p:
                nn.init.constant_(q, 0)

    def to(self, device):
        """
        Override the `to` method to ensure all parameters and buffers are moved to the specified device.
        """
        super().to(device)
        self.weights = self.weights.to(device)
        self.device = device
        return self
    
    def encode(self, graph, ckey='cell', rkey='region'):
        c_feat, r_feat = self.cell_feature, self.region_feature
        c_hidden, r_hidden = self.weights[0] * c_feat, self.weights[0] * r_feat
        for w, encoder in zip(self.weights[1:], self.encoders):
            c_feat, r_feat = encoder(graph, c_feat, r_feat, ckey, rkey)
            c_hidden = c_hidden + w * c_feat
            r_hidden = r_hidden + w * r_feat
        
        return c_hidden, r_hidden, c_feat, r_feat
    
    def decode(self, pos_graph, neg_graph, c_feat, r_feat,rs_factor, ckey='cell',rkey='region'):
        pos_pre = self.decoder.forward(pos_graph, c_feat, r_feat, ckey, rkey)
        neg_pre = self.decoder.forward(neg_graph, c_feat, r_feat, ckey, rkey)
        return pos_pre, neg_pre
    
    def compute_bpr_loss(self, pos_graph, neg_graph, c_feat, r_feat):
        pos_scores = self.DotDecoder(pos_graph, c_feat, r_feat).sum(dim=1)
        neg_scores = self.DotDecoder(neg_graph, c_feat, r_feat).sum(dim=1)

        bpr_loss = torch.mean(-torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-9))
        return bpr_loss
    
    def inference(self, enc_graph, pos_graph, neg_graph=None, ckey='cell',rkey='region'):
        """
        bpr + KL + NB loss
        """
        c_feat, r_feat, c_last, r_last = self.encode(enc_graph, ckey, rkey)
        # calculate NB loss
        pos_pre, neg_pre = self.decode(pos_graph, neg_graph, c_feat, r_feat, self.region_factors, ckey, rkey)
        # calculate bpr loss
        bpr_loss = self.compute_bpr_loss(pos_graph, neg_graph, c_feat, r_feat)

        return bpr_loss, pos_pre, neg_pre

class VA_BRAID(nn.Module):
    def __init__(
        self,
        n_layers,
        n_cells,
        n_regions,
        drop_out,
        feats_dim,
        learnable_weight=False,
        decoder='NB',
        use_cell2cell = False
        ):
        super().__init__()
        """a novel scATAC-seq representation learning method based on graph node embedding
        ---------------------------------------------------------------------------------------
        n_layers : int
            number of GCN layers
        n_cells : int
            number of cells
        n_regions : int
            number of regions   
        drop_out : float
            dropout rate (neighbors)
        feats_dim : int
            node feature dimension
        learnable_weight : boolean
            whether to learn weights for embedding aggregation, if False, use 1/n_layers
        """
        self.n_cells = n_cells
        self.n_regions = n_regions

        self.cell_feature = nn.Parameter(torch.Tensor(self.n_cells, feats_dim))
        self.region_feature = nn.Parameter(torch.Tensor(self.n_regions, feats_dim))

        nn.init.xavier_uniform_(self.cell_feature)
        nn.init.xavier_uniform_(self.region_feature)

        self.cell_factors = torch.nn.Parameter(torch.ones(self.n_cells))
        self.region_factors = torch.nn.Parameter(torch.ones(self.n_regions))

        self.n_layers = n_layers
        self.encoders = nn.ModuleList()
        for _ in range(n_layers):
            self.encoders.append(lightgraphconvlayer(drop_out=drop_out, use_cell2cell=use_cell2cell))
            # self.encoders.append(GAT(in_feats=feats_dim, n_heads=2))

        if self.n_layers == 2:
            self.weights = torch.tensor([1., 1. / 2, 1. / 2])
        else:
            self.weights = torch.ones([self.n_layers + 1, 1]) / (self.n_layers + 1)

        if learnable_weight:
            self.weights = nn.Parameter(self.weights)
        if decoder =='Poisson':
            self.decoder = PoissonDecoder(feats_dim)
        elif decoder =='NB':
            self.decoder = NBDecoder(feats_dim)
        elif decoder == 'Dot':
            self.decoder = DotDecoder()

        self.eps_weight = torch.nn.Parameter(torch.randn(feats_dim, feats_dim), requires_grad=True)
        self.eps_bias = torch.nn.Parameter(torch.zeros(feats_dim), requires_grad=True)

        self.device = None

        for p, q in self.decoder.named_parameters():
            if 'weight' in p:
                nn.init.kaiming_normal_(q)
            elif 'bias' in p:
                nn.init.constant_(q, 0)

        self.l2_normalize = True
        self.temp_cluster = 0.1
        self.temp_node = 0.5


    def to(self, device):
        """
        Override the `to` method to ensure all parameters and buffers are moved to the specified device.
        """
        super().to(device)
        self.weights = self.weights.to(device)
        self.device = device
        return self
    
    def encode(self, graph, ckey='cell', rkey='region'):
        c_feat, r_feat = self.cell_feature, self.region_feature
        c_hidden, r_hidden = self.weights[0] * c_feat, self.weights[0] * r_feat
        for w, encoder in zip(self.weights[1:], self.encoders):
            c_feat, r_feat = encoder(graph, c_feat, r_feat, ckey, rkey)
            c_hidden = c_hidden + w * c_feat
            r_hidden = r_hidden + w * r_feat
        
        c_logstd = torch.matmul(c_hidden, self.eps_weight) + self.eps_bias
        r_logstd = torch.matmul(r_hidden, self.eps_weight) + self.eps_bias

        c_std = torch.exp(c_logstd)
        r_std = torch.exp(r_logstd)

        
        return c_hidden, r_hidden, c_std, r_std
    
    def decode(self, pos_graph, neg_graph, c_feat, r_feat, ckey='cell', rkey='region'):
        # pos_pre = self.decoder.forward2(pos_graph, c_feat, r_feat, self.cell_factors, self.region_factors, ckey, rkey)
        # neg_pre = self.decoder.forward2(neg_graph, c_feat, r_feat, self.cell_factors, self.region_factors, ckey, rkey)
    
        pos_pre = self.decoder.forward1(pos_graph, c_feat, r_feat, self.region_factors, ckey, rkey)
        neg_pre = self.decoder.forward1(neg_graph, c_feat, r_feat, self.region_factors, ckey, rkey)

        # pos_pre = self.decoder.forward(pos_graph, c_feat, r_feat, ckey, rkey)
        # neg_pre = self.decoder.forward(neg_graph, c_feat, r_feat, ckey, rkey)
        return pos_pre, neg_pre
    
    def compute_bpr_loss(self, pos_graph, neg_graph, c_feat, r_feat):
        pos_scores = self.DotDecoder(pos_graph, c_feat, r_feat).sum(dim=1)
        neg_scores = self.DotDecoder(neg_graph, c_feat, r_feat).sum(dim=1)

        bpr_loss = torch.mean(-torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-9))
        return bpr_loss
    
    def kl_regularizer(self, mean, std):
        '''
        KL term in ELBO loss
        Constraint approximate posterior distribution closer to prior
        '''
        regu_loss = -0.5 * (1 + 2 * std - torch.square(mean) - torch.square(torch.exp(std)))
        return regu_loss.sum(1).mean()
    
    def reparameter(self, mean, std, scale=0.001):
        # simulation, depth, clustering 0.001

        random_noise = torch.randn(std.shape).to(self.device)
        embedding = mean + std * random_noise * scale
        return embedding
    
    def inference(self, enc_graph, pos_graph, neg_graph=None, ckey='cell',rkey='region'):
        """
        bpr + KL + NB loss
        """
        c_mean, r_mean, c_std, r_std = self.encode(enc_graph, ckey, rkey)
        # sampling
        c_feat = self.reparameter(c_mean, c_std)
        r_feat = self.reparameter(r_mean, r_std)

        # calculate NB loss
        pos_pre, neg_pre = self.decode(pos_graph, neg_graph, c_feat, r_feat, ckey, rkey)
        loss_access = self.decoder.loss(pos_pre, pos_graph.edata['count'])
        loss_unaccess = self.decoder.loss(neg_pre)
        # calculate bpr loss
        # bpr_loss = self.compute_bpr_loss(pos_graph, neg_graph, c_feat, r_feat)
        # n_edges = pos_pre[2].shape[0]
        # bpr_loss = (1 - pos_pre[2].unsqueeze(1) + neg_pre[2].view(n_edges, -1)).clamp(min=0).mean()
        # hinge_auc_loss
        # bpr_loss = (torch.square(torch.clamp(1 - (pos_pre[-1] - neg_pre[-1]), min=0))).mean()
        bpr_loss = torch.mean(-torch.log(torch.sigmoid(pos_pre[-1] - neg_pre[-1]) + 1e-9))
        # calculate kl loss
        kl_loss = self.kl_regularizer(torch.cat((c_mean, r_mean), 0), torch.cat((c_std, r_std), 0))

        # reg loss
        reg_loss = (self.cell_feature[pos_graph.edges()[0]].norm(2).pow(2) +
                        self.region_feature[pos_graph.edges()[1]].norm(2).pow(2) +
                        self.region_feature[neg_graph.edges()[1]].norm(2).pow(2)) / float(pos_graph.num_edges())
        

        # return bpr_loss, kl_loss/float(pos_graph.num_edges()), loss_access+loss_unaccess, reg_loss
        return bpr_loss, kl_loss/float(pos_graph.num_edges()), (loss_access+loss_unaccess), reg_loss
    
    
    def inference1(self, enc_graph, pos_graph, neg_graph=None, ckey='cell',rkey='region', cell_2cluster=None):
        id = pos_graph.edges()
        cells = torch.unique(id[0])
        pos_regions = torch.unique(id[1])

        c_mean, r_mean, c_std, r_std = self.encode(enc_graph, ckey, rkey)
        # sampling
        c_feat1 = self.reparameter(c_mean, c_std)
        r_feat1 = self.reparameter(r_mean, r_std)
        c_feat2 = self.reparameter(c_mean, c_std)
        r_feat2 = self.reparameter(r_mean, r_std)

        # calculate NB loss
        pos_pre, neg_pre = self.decode(pos_graph, neg_graph, c_feat1, r_feat1, self.region_factors, ckey, rkey)
        loss_access = NB_Loss(pos_graph.edata['count'], pos_pre[0], pos_pre[1])
        loss_unaccess = NB_Loss(torch.zeros_like(neg_pre[0]), neg_pre[0], neg_pre[1])
        # calculate bpr loss
        bpr_loss = torch.mean(-torch.log(torch.sigmoid(pos_pre[2] - neg_pre[2]) + 1e-9))
        # calculate kl loss
        kl_loss = self.kl_regularizer(torch.cat((c_mean, r_mean), 0), torch.cat((c_std, r_std), 0))

        # reg loss
        reg_loss = (self.cell_feature[pos_graph.edges()[0]].norm(2).pow(2) +
                        self.region_feature[pos_graph.edges()[1]].norm(2).pow(2) +
                        self.region_feature[neg_graph.edges()[1]].norm(2).pow(2)) / float(pos_graph.num_edges())
        # calculate node-level contrastive loss
        cl_loss_node = self.compute_cl_loss_node(c_feat1, r_feat1, c_feat2, r_feat2, cells, pos_regions)
        # calculate cluster-level contrastive loss
        cl_loss_cluster = self.compute_cl_loss_cluster(c_feat1, c_feat2, cells, cell_2cluster)

        return bpr_loss, kl_loss/float(pos_graph.num_edges()), loss_access+loss_unaccess, reg_loss, cl_loss_node, cl_loss_cluster

    def compute_cl_loss_node(self, c_feat1, r_feat1, c_feat2, r_feat2, cells, pos_regions):

        ### cell part ###
        cell_sub1, cell_sub2 = c_feat1[cells], c_feat2[cells]
        cell_sub1 = F.normalize(cell_sub1, p=2, dim=1)
        cell_sub2 = F.normalize(cell_sub2, p=2, dim=1)
        pos_score_cell = torch.multiply(cell_sub1, cell_sub2).sum(1) # [bs, 1]
        all_score_cell = torch.matmul(cell_sub1, cell_sub2.transpose(0, 1)) # [bs, bs]
        pos_score_cell = torch.exp(pos_score_cell / self.temp_node) # [bs, 1]
        all_score_cell = torch.exp(all_score_cell / self.temp_node).sum(1) # [bs, 1]
        cl_loss_cell = -torch.log(pos_score_cell / all_score_cell).mean()

        ### region part ###
        region_sub1, region_sub2 = r_feat1[pos_regions], r_feat2[pos_regions]
        region_sub1 = F.normalize(region_sub1, p=2, dim=1)
        region_sub2 = F.normalize(region_sub2, p=2, dim=1)
        pos_score_region = torch.multiply(region_sub1, region_sub2).sum(1)
        all_score_region = torch.matmul(region_sub1, region_sub2.transpose(0, 1))
        pos_score_region = torch.exp(pos_score_region / self.temp_node)
        all_score_region = torch.exp(all_score_region / self.temp_node).sum(1)
        cl_loss_region = -torch.log(pos_score_region / all_score_region).mean()
        cl_loss_node = cl_loss_cell + cl_loss_region
        return cl_loss_node
    
    def compute_cl_loss_cluster(self, c_feat1, c_feat2, cells, cell_2cluster):
        '''
        Cluster-level contrastive learning
        (1) K-means clustering as a special instance that prototype distribution is onehot
        (2) We select cells with a same clustering prototype as the positive samples for each anchor node
        (3) Contrastive temperature can be assigned a smaller value compared to node-level cl loss
        '''
        ###  pos samples  ###

        cell_cluster_id = cell_2cluster[cells]
        cell_mask = torch.eq(cell_cluster_id, cell_cluster_id.transpose(0, 1)).float()
        avg_positive_cell = cell_mask.sum(dim=1)

        # item_cluster_id = item_2cluster[items]
        # item_mask = torch.eq(item_cluster_id, item_cluster_id.transpose(0,1)).float()  # [bs, bs]
        # avg_positive_item = item_mask.sum(dim=1)

        ###  user contrastive learning  ###
        cell_sub1, cell_sub2 = c_feat1[cells], c_feat2[cells]
        cell_sub1 = F.normalize(cell_sub1, p=2, dim=1)
        cell_sub2 = F.normalize(cell_sub2, p=2, dim=1)
        logit = torch.matmul(cell_sub1, cell_sub2.transpose(0, 1))
        logit = logit - logit.detach().max(dim=1, keepdim=True)[0]  # remove max value of each row
        exp_logit = torch.exp(logit / self.temp_cluster)
        pos_score_cell = (exp_logit * cell_mask).sum(1) / avg_positive_cell
        all_score_cell = exp_logit.sum(1)
        cl_loss_cell = -torch.log(pos_score_cell / all_score_cell).mean()

        return cl_loss_cell
    
    @torch.no_grad()
    def run_PyTorch_kmeans(self, x, num_cluster):
        kwargs = {
            'metric': 'euclidean' if self.l2_normalize else 'euclidean',
            'distributed': False,
            'random_state': 0,
            'n_clusters': num_cluster,
            'verbose': False
        }
        clustering_model = PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)

        psedo_labels = clustering_model.fit_predict(x)
        cluster_centers = clustering_model.cluster_centers_

        return psedo_labels.unsqueeze(1).to(self.device)
    
    def cluster_step(self, num_cluster):
        cell_2cluster = self.run_PyTorch_kmeans(self.cell_feature, num_cluster)

        return cell_2cluster
    
    
   