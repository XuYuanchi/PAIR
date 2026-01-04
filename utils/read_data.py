import scanpy as sc
from glob import glob
import pandas as  pd
from anndata import AnnData
import os
import scipy.sparse as sp
from typing import Optional
import numpy as np

def read_mtx(path):
    """
    Read mtx format data folder including: 
        matrix file: e.g. count.mtx or matrix.mtx
        barcode file: e.g. barcode.txt
        feature file: e.g. feature.txt
    """
    for filename in glob(path+'/*'):
        if ('count' in filename or 'matrix' in filename or 'data' in filename) and ('mtx' in filename):
            adata = sc.read_mtx(filename).T
    for filename in glob(path+'/*'):
        if 'barcode' in filename:
            barcode = pd.read_csv(filename, sep='\t', header=None).iloc[:, -1].values
            print(len(barcode), adata.shape[0])
            if len(barcode) != adata.shape[0]:
                adata = adata.transpose()
            adata.obs = pd.DataFrame(index=barcode)
        if 'gene' in filename or 'peaks' in filename or 'feature' in filename:
            gene = pd.read_csv(filename, sep='\t', header=None).iloc[:, -1].values
            if len(gene) != adata.shape[1]:
                adata = adata.transpose()
            adata.var = pd.DataFrame(index=gene)
    return adata

def load_file(path):  
    """
    Load single cell dataset from file
    """
    if os.path.exists(path+'.h5ad'):
        adata = sc.read_h5ad(path+'.h5ad')
    elif os.path.isdir(path): # mtx format
        adata = read_mtx(path)
    elif os.path.isfile(path):
        if path.endswith(('.csv', '.csv.gz')):
            adata = sc.read_csv(path).T
        elif path.endswith(('.txt', '.txt.gz', '.tsv', '.tsv.gz')):
            df = pd.read_csv(path, sep='\t', index_col=0).T
            adata = AnnData(df.values, dict(obs_names=df.index.values), dict(var_names=df.columns.values))
        elif path.endswith('.h5ad'):
            adata = sc.read_h5ad(path)
    # elif path.endswith(tuple(['.h5mu/rna', '.h5mu/atac'])):
    #     import muon as mu
    #     adata = mu.read(path)
    else:
        raise ValueError("File {} not exists".format(path))
        
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    adata.var_names_make_unique()
    return adata


def load_files(root):
    """
    Load single cell dataset from files
    """
    print(root)
    if root.split('/')[-1] == '*':
        adata = []
        for root in sorted(glob(root)):
            adata.append(load_file(root))
        return AnnData.concatenate(*adata, batch_key='sub_batch', index_unique=None)
    else:
        return load_file(root)


def concat_data(
        data_list, 
        batch_categories=None, 
        join='inner',             
        batch_key='batch', 
        index_unique=None, 
        save=None
    ):
    """
    Concat multiple datasets
    """
    if len(data_list) == 1:
        return load_files(data_list[0])
    elif isinstance(data_list, str):
        return load_files(data_list)
    adata_list = []
    for root in data_list:
        adata = load_files(root)
        adata_list.append(adata)
        
    if batch_categories is None:
        batch_categories = list(map(str, range(len(adata_list))))
    else:
        assert len(adata_list) == len(batch_categories)
    [print(b, adata.shape) for adata,b in zip(adata_list, batch_categories)]
    concat = AnnData.concatenate(*adata_list, join=join, batch_key=batch_key,
                                batch_categories=batch_categories, index_unique=index_unique)  
    if save:
        concat.write(save, compression='gzip')
    return concat

def reads_to_fragments(
    adata: AnnData,
    read_layer: Optional[str] = None,
    fragment_layer: str = "fragments",
) -> None:
    """Convert scATAC-seq read counts to appoximate fragment counts.

    Parameters
    ----------
    adata
        AnnData object that contains read counts.
    read_layer
        Key in`.layer` that the read counts are stored in.
    fragment_layer
        Key in`.layer` that the fragment counts will be stored in.

    Returns
    -------
    Adds layer with fragment counts in `.layers[fragment_layer]`.
    """
    adata.layers[fragment_layer] = (
        adata.layers[read_layer].copy() if read_layer else adata.X.copy()
    )
    if sp.issparse(adata.layers[fragment_layer]):
        adata.layers[fragment_layer].data = np.ceil(adata.layers[fragment_layer].data / 2)
    else:
        adata.layers[fragment_layer] = np.ceil(adata.layers[fragment_layer] / 2)

def load_data(path, config):

    adata = concat_data(path)

    print('Raw dataset shape: {}'.format(adata.shape))

    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)

    binary_data = (adata.X > 0).astype(int)
    adata.layers['binary'] = binary_data

    # if adata.raw is not None:
    #     adata.X = adata.raw.X

    sc.pp.filter_cells(adata, min_genes=config.min_regions)

    if config.min_cells < 1:
        min_cells = config.min_cells * adata.shape[0]
    else:
        min_cells = config.min_cells
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # read count to fragment
    if (adata.X == 1).sum() < (adata.X == 2).sum():
        reads_to_fragments(adata)
    else:
        adata.layers['fragments'] = adata.X.copy()

    print('Processing dataset shape:{}'.format(adata.shape))

    # Calculate the cell library size factor
    adata.layers['normalized'] = adata.layers['fragments'].copy()
    # adata.layers['normalized'] = adata.X.copy()
    sc.pp.normalize_total(adata,layer='normalized', key_added='n_counts')
    adata.obs['cs_factor'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    # Log Normalization
    sc.pp.log1p(adata,layer='normalized')
    # Calculate the gene size factor
    # adata.var['rs_factor'] = np.max(adata.layers['normalized'], axis=0, keepdims=True).reshape(-1)
    adata.var['rs_factor'] = adata.layers['normalized'].max(axis=0).toarray().flatten()

    return adata