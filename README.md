# PAIR: Reconstructing Single-Cell Open-Chromatin Landscapes for Transcription Factor Regulome Mapping
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
This repository contains code and data to support data analyses and reproduce results from the paper PAIR: Reconstructing Single-Cell Open-Chromatin Landscapes for Transcription Factor Regulome Mapping.
- [Abstract](#abstract)
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Usage](#Usage)
- [Data Availability](#data-availability)
- [License](#license)

# Abstract
Single-cell ATAC-seq (scATAC-seq) enables the interrogation of chromatin accessibility at cellular resolution, yet its practical utility is often constrained by limited sequencing depth, extreme sparsity, and pervasive technical missingness, which collectively hamper robust cell-state delineation and inference of transcription factor (TF) regulatory programs. We present PAIR, a probabilistic framework that restores scATAC-seq accessibility profiles by directly modeling the native cell–peak bipartite structure of chromatin accessibility. PAIR leverages a bipartite graph encoder to learn representations for both cells and peaks, and incorporates a variational latent layer to explicitly capture uncertainty arising from sparse and noisy measurements. To jointly recover discrete accessibility patterns and quantitative signal, PAIR integrates two complementary decoders: a qualitative decoder that reconstructs open/closed cell–peak incidences and a quantitative decoder that models accessibility counts under a Negative Binomial likelihood. Trained end-to-end with variational and embedding regularization, PAIR yields cell and peak embeddings and an imputed accessibility matrix that improves downstream analyses. Across simulated datasets with controlled sequencing depth, noise, and dropout, as well as multiple publicly available benchmarks, PAIR consistently improves clustering performance and increases sensitivity for differential accessibility. Beyond cell-level analyses, PAIR-derived peak embedding enable locus-centric regulatory interrogation: co-accessibility analysis around SOX10 reveals structured regulatory neighborhoods, and graph-based peak modules show selective activity across melanoma cell states and identify gene sets with clinically relevant survival associations. In a forebrain atlas, PAIR restores regulatory signals spanning both promoter-proximal and distal elements and uncovers biologically coherent enrichment patterns consistent with neuronal specialization.

# Overview
<div align=center>
<img src="https://github.com/XuYuanchi/PAIR/blob/main/framework.png" height="800" width="1000">
</div>

# System Requirements
## Hardware requirements
`PAIR` requires only a standard computer with enough RAM to support the in-memory operations.

## Software requirements
### OS Requirements
This package is supported for *Linux*. The package has been tested on the following systems:
+ Linux: Ubuntu 18.04

### Python Dependencies
`PAIR` mainly depends on the Python scientific stack.
```
numpy
scipy
PyTorch
scikit-learn
pandas
scanpy
anndata
```
For a specific setting, please see <a href="https://github.com/XuYuanchi/PAIR/blob/main/environment.yml">requirement</a>.

# Installation Guide
```
$ git clone https://github.com/XuYuanchi/PAIR.git
$ conda create -n pair python=3.9.19
$ conda activate pair
$ conda env create -f environment.yml
```

# Usage
`PAIR` is a bipartite graph-based autoencoder model for scATAC-seq data analysis, such as imputation and clustering. 
The example can be seen in the <a href="https://github.com/XuYuanchi/PAIR/blob/main/main.py">train.py</a>.

# Data Availability
The data that support the findings of this study are openly available in <a href="https://doi.org/10.5281/zenodo.18085474">Zenodo</a>

# License
This project is covered under the **MIT License**.

# Citation

```
@article{su2024distribution,
  title={PAIR: Reconstructing Single-Cell Open-Chromatin Landscapes for Transcription Factor Regulome Mapping},
  author={Su Yanchi et al.},
  journal={Advanced Science},
  pages={e24392},
  year={2026},
  publisher={wiley}
}
```
