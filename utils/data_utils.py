import dgl
import numpy as np
import pandas as pd
import scanpy as sc
import torch

import argparse
import subprocess
import random
import os

import scipy.sparse as sp
import warnings
from sklearn.feature_extraction.text import TfidfTransformer

from .graph import construct_region_graph, add_degree, build_spatial_cell_edges


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        #add paired parameters into namespace
        setattr(namespace, key, new_value)
    return namespace

def setup_seed(seed):
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# get gpu usage
def get_gpu_memory_map():
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    # gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory

def auto_select_gpu(memory_threshold = 7000, smooth_ratio=200, strategy='greedy'):
    gpu_memory_raw = get_gpu_memory_map() + 10
    if strategy=='random':
        gpu_memory = gpu_memory_raw/smooth_ratio
        gpu_memory = gpu_memory.sum() / (gpu_memory+10)
        gpu_memory[gpu_memory_raw>memory_threshold] = 0
        gpu_prob = gpu_memory / gpu_memory.sum()
        cuda = str(np.random.choice(len(gpu_prob), p=gpu_prob))
        print('GPU select prob: {}, Select GPU {}'.format(gpu_prob, cuda))
    elif strategy == 'greedy':
        cuda = np.argmin(gpu_memory_raw)
        print('GPU mem: {}, Select GPU {}'.format(gpu_memory_raw[cuda], cuda))
    return cuda


# def preprocess(adata, filter_min_counts=True, size_factors=True, normalize_input=False, logtrans_input=True):
#     if size_factors or normalize_input or logtrans_input:
#         adata.raw = adata.copy()
#     else:
#         adata.raw = adata

#     if filter_min_counts:
#         sc.pp.filter_genes(adata, min_cells=3)
#         sc.pp.filter_cells(adata, min_genes=200)

#     if size_factors:
#         sc.pp.normalize_per_cell(adata)
#         adata.obs['cs_factor'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
#     else:
#         adata.obs['cs_factor'] = 1.0

#     if logtrans_input:
#         sc.pp.log1p(adata)

#     gs_factor = np.max(adata.X, axis=0, keepdims=True)
#     adata.var['rs_factor'] = gs_factor.reshape(-1)

#     if normalize_input:
#         sc.pp.scale(adata)

#     return adata
def tf_idf_transform(data):
    model = TfidfTransformer(smooth_idf=False, norm="l2")
    model = model.fit(np.transpose(data))
    model.idf_ -= 1
    tf_idf = np.transpose(model.transform(np.transpose(data)))

    return tf_idf


def make_graph(adata, raw_exp=True, region_similarity=False, cell_similarity=False):
    # weight = tf_idf_transform(adata.layers['binary'].T)
    # weight = weight.T.tocoo().data.reshape(-1, 1)
    X = adata.layers['binary'].tocoo()
    # X = adata.X.tocoo()
    num_cells, num_regions = X.shape

    # Make open/train graph
    num_nodes_dict = {'cell': num_cells, 'region': num_regions}
    open_train_cell, open_train_region = X.row, X.col
    unopen_edges = np.where(X.todense() == 0)

    # expression edges
    open_edge_dict = {
        ('cell', 'access', 'region'): (open_train_cell, open_train_region),
        ('region', 'reverse-access', 'cell'): (open_train_region, open_train_cell)
    }

    coopen_edges, uncoopen_edges = None, None
    if region_similarity:
        coopen_edges, uncoopen_edges = construct_region_graph(X)
        open_edge_dict[('region', 'co-access', 'region')] = coopen_edges

    if cell_similarity:
        rad_cutoff=None,
        rad_coef=1.5,
        coor_key='spatial',
        include_self=False,
        spatial_edges, spatial_radius = build_spatial_cell_edges(
            adata,
            rad_cutoff=rad_cutoff,
            rad_coef=rad_coef,
            coor_key=coor_key,
            include_self=include_self,
        )
        open_edge_dict[('cell', 'spatial-neighbor', 'cell')] = spatial_edges


    # open encoder/decoder graph
    enc_graph = dgl.heterograph(open_edge_dict, num_nodes_dict=num_nodes_dict)
    # enc_graph.edges['access'].data['weight'] = torch.Tensor(weight.data.reshape(-1, 1))
    # enc_graph.edges['reverse-access'].data['weight'] = torch.Tensor(weight.data.reshape(-1, 1))

    open_edge_dict.pop(('region', 'reverse-access', 'cell'))
    dec_graph = dgl.heterograph(open_edge_dict, num_nodes_dict=num_nodes_dict)

    # add degree to cell/region nodes
    add_degree(enc_graph, ['access'] + (['co-access'] if region_similarity else []))

    # If use Poisson decoder, add size factor to cell/region nodes
    if raw_exp:
        # Raw = pd.DataFrame(adata.layers['fragments'], index=list(adata.obs_names), columns=list(adata.var_names))
        # X = Raw[list(adata.var_names)].values
        # open_value = X[open_train_cell, open_train_region].reshape(-1, 1)
        # open_value = weight
        open_value = X.data.reshape(-1, 1)   
        dec_graph.nodes['cell'].data['cs_factor'] = torch.Tensor(adata.obs['cs_factor'].values).reshape(-1, 1)
        dec_graph.nodes['region'].data['rs_factor'] = torch.Tensor(adata.var['rs_factor'].values).reshape(-1, 1)

    else:
        ## Deflate the edge values of the bipartite graph to between 0 and 1
        X = X / adata.var['rs_factor'].values
        exp_value = X[open_train_cell, open_train_region].reshape(-1, 1)

    data_dict = dict(
        adata=adata, 
        open_value = open_value, 
        enc_graph = enc_graph, 
        dec_graph = dec_graph, 
        unopen_edges = unopen_edges, 
        coopen_edges = coopen_edges, 
        uncoopen_edges = uncoopen_edges,
        # weight = weight
    )

    return data_dict


def run_tfidf(
    object,
    method=1,
    scale_factor=1e4,
    idf=None,
    verbose=True,
):
    if isinstance(object, np.ndarray):
        object = sp.csr_matrix(object)
    elif not isinstance(object, sp.csr_matrix):
        object = sp.csr_matrix(object)

    if verbose:
        print("Performing TF-IDF normalization")

    npeaks = np.array(object.sum(axis=1)).flatten()
    if np.any(npeaks == 0):
        warnings.warn("Some cells contain 0 total counts")

    if method == 4:
        tf = object
    else:
        # Normalize term frequencies
        tf = sp.diags(1 / npeaks) @ object

    precomputed_idf = False
    if idf is not None:
        precomputed_idf = True
        if not isinstance(idf, np.ndarray):
            raise ValueError("idf parameter must be a numeric vector")
        if len(idf) != object.shape[0]:
            raise ValueError("Length of supplied IDF vector does not match the number of rows in input matrix")
        if np.any(idf == 0):
            raise ValueError("Supplied IDF values cannot be zero")
        if verbose:
            print("Using precomputed IDF vector")
    else:
        rsums = np.array(object.sum(axis=0)).flatten()
        if np.any(rsums == 0):
            warnings.warn("Some features contain 0 total counts")
        idf = object.shape[0] / rsums
        # idf = np.log1p(object.shape[1] / rsums) if method == 2 else object.shape[1] / rsums
    if method == 2:
        idf = np.log(1+idf)
    elif method == 3:
        tf.data = np.log1p(tf.data * scale_factor)
        if idf is None:
            idf = np.log1p(idf)

    norm_data = tf @ sp.diags(idf)

    if method == 1:
        norm_data.data = np.log1p(norm_data.data * scale_factor)

    norm_data = sp.csr_matrix(norm_data)

    # 设置NA值为0
    norm_data.data[np.isnan(norm_data.data)] = 0

    return norm_data
