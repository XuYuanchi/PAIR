import numpy as np
import scanpy as sc
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import math
import torch
import torch.nn as nn


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss


def pearsonr_error(y, h):
    res = []
    if len(y.shape) < 2:
        y = y.reshape((1, -1))
        h = h.reshape((1, -1))

    for i in range(y.shape[0]):
        res.append(pearsonr(y[i], h[i])[0])
    return np.mean(res)


def cosine_similarity_score(y, h):
    if len(y.shape) < 2:
        y = y.reshape((1, -1))
        h = h.reshape((1, -1))
    cos = cosine_similarity(y, h)
    res = []
    for i in range(len(cos)):
        res.append(cos[i][i])
    return np.mean(res)

def kmeans(adata, n_clusters, use_rep=None):
    k_means = KMeans(n_clusters, n_init=20, random_state=0)
    y_pred = k_means.fit_predict(adata.obsm[use_rep])
    adata.obs['kmeans'] = y_pred
    adata.obs['kmeans'] = adata.obs['kmeans'].astype(str).astype('category')
    return adata

def louvain(adata, resolution=None, use_rep=None):
    sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.louvain(adata, resolution=resolution)
    return adata


def getNClusters(adata, n_cluster, range_min=0, range_max=3, max_steps=50, method='louvain', use_rep=None):
    """
    Function will test different settings of louvain to obtain the target number of clusters.
    credit to Lucas Pinello lab. See: https://github.com/pinellolab/scATAC-benchmarking

    It can get cluster for both louvain and leiden.
    You can specify the obs variable name as key_added. 
    """
    if n_cluster is None:
        this_step = max_steps - 1
    else:
        this_step = 0
    this_min = float(range_min)
    this_max = float(range_max)

    sc.pp.neighbors(adata, n_neighbors=20, use_rep=use_rep)

    while this_step < max_steps:
        this_resolution = this_min + ((this_max-this_min)/2)

        if method == 'louvain':
            sc.tl.louvain(adata, resolution=this_resolution)
            this_clusters = adata.obs['louvain'].nunique()
        elif method == 'leiden':
            sc.tl.leiden(adata, resolution=this_resolution, flavor="igraph", n_iterations=2)
            this_clusters = adata.obs['leiden'].nunique()

        if this_clusters > n_cluster:
            this_max = this_resolution
        elif this_clusters < n_cluster:
            this_min = this_resolution
        else:
            return adata
        this_step += 1

    print('cannot find the number of clusters')
    print('Clustering solution from last iteration is used: ' + str(this_clusters) + ' at resolution' + str(this_resolution))
    return adata

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.float64).astype(np.int64)
    y_pred = y_pred.astype(np.float64).astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size


def calculate_metric(pred, label):
    nmi = np.round(metrics.normalized_mutual_info_score(label, pred), 4)
    ari = np.round(metrics.adjusted_rand_score(label, pred), 4)

    return nmi, ari


def ZINB_Loss(mean, disp, pi, x=None, eps=1e-10):
    if x is None:
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        return zero_case.mean()

    t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
    t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
    nb_final = t1 + t2

    nb_case = nb_final - torch.log(1.0 - pi + eps)

    return nb_case.mean()

def NB_Loss(x, mu, theta, eps=1e-8):
    """
    Note: All inputs should be torch Tensors
    log likelihood (scalar) of a minibatch according to a nb model.

    Variables:
    mu: mean of the negative binomial (has to be positive support) (shape: minibatch x genes)
    theta: inverse dispersion parameter (has to be positive support) (shape: minibatch x genes)
    eps: numerical stability constant
    """
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))  # In this case, we reshape theta for broadcasting

    log_theta_mu_eps = torch.log(theta + mu + eps)

    res = theta * (torch.log(theta + eps) - log_theta_mu_eps) + \
        x * (torch.log(mu + eps) - log_theta_mu_eps) + \
        torch.lgamma(x + theta) - \
        torch.lgamma(theta) - \
        torch.lgamma(x + 1)

    return -res.mean()

