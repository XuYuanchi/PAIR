import gc
import torch
import torch.nn as nn
from torch.distributions import Poisson
import dgl
import dgl.graphbolt as gb
from functools import partial

import numpy as np
import wandb
# import cupy as cp
from sklearn import preprocessing

from utils import make_graph, calculate_metric, kmeans, getNClusters, ZINB_Loss
from .BRAID import BRAID, VA_BRAID

import time
import pickle



class Trainer(nn.Module):
    def __init__(
            self, 
            config,
            sample_rate: float = 0.1,
            # region_similarity: bool = False,
            verbose: bool = True,
            use_rep:str = 'feat',
            impute: bool = False,
            save_model: bool = False,
            ):
        super().__init__()

        self.config = config
        self.sample_rate = sample_rate
        # self.region_similarity = region_similarity
        self.verbose = verbose
        self.log_interval = config.log_interval
        self.cl_type = config.cl_type
        self.use_rep = use_rep
        self.impute = impute
        self.save_model = save_model

    def prepare_data(self, adata):


        if self.cl_type:
            le = preprocessing.LabelEncoder()
            self.cell_type = le.fit_transform(adata.obs[self.cl_type])
        else:
            None
        self.n_cells, self.n_regions = adata.X.shape

        if self.cell_type is not None:
            self.n_clusters = len(np.unique(self.cell_type))

        try:
            with open('/home/suyanchi/project/atac/temp/' + self.config.dataset + '.pkll', 'rb') as f:
                all_data = pickle.load(f)
            # all_data = np.load('/home/suyanchi/project/atac/temp/' + self.config.dataset + '.npy').item()
                print("successfully loaded...")
        except:
            all_data = make_graph(adata)
            with open('/home/suyanchi/project/atac/temp/' + self.config.dataset + '.pkl', 'wb') as f:
                pickle.dump(all_data, f, pickle.HIGHEST_PROTOCOL)
            # np.save('/home/suyanchi/project/atac/temp/' + self.config.dataset + '.npy', all_data)
        return all_data

    def train(self, adata):

        all_data = self.prepare_data(adata)


        # make dgl graph object
        enc_graph = all_data['enc_graph']
        # make sampling graph
        sampling_graph = gb.from_dglgraph(enc_graph)
        # make itemset
        

        datapipe = gb.ItemSampler(itemset, batch_size=1024, shuffle=True)
        # sampling negtive graph
        datapipe = datapipe.sample_uniform_negative(sampling_graph, 5)
        datapipe = datapipe.sample_neighbor(sampling_graph, [10, 10]) # 2 layers.
        datapipe = datapipe.transform(gb.exclude_seed_edges)

        # feature TorchBasedFeatureStore
        # graph FusedCSCSamplingGraph
        # train_set dgl.graphbolt.itemset.ItemSet
        datapipe = datapipe.fetch_feature(
            feature,
            node_feature_keys={"user": ["feat"], "item": ["feat"]}
        )
        datapipe = datapipe.copy_to(device)
        dataloader = gb.DataLoader(datapipe)


        exclude_seed_edges = partial(
            gb.exclude_seed_edges,
            include_reverse_edges=True,
            reverse_etypes_mapping={
                "user:like:item": "item:liked_by:user",
                "user:follow:user": "user:followed_by:user",
                },
            )
        datapipe = datapipe.transform(exclude_seed_edges)

        #######################   Record values   #######################
        all_loss = []

        best_ari_k, best_ari_l = 0, 0
        best_nmi_k, best_nmi_l = 0, 0
        all_ari_k, all_ari_l = [], []
        all_nmi_k, all_nmi_l = [], []

        best_iter_k, best_iter_l = -1, -1

        #######################   train/test data   #######################

        n_pos_edges, n_neg_edges = int(self.sample_rate * len(all_data['open_value'])), int(self.sample_rate * len(all_data['open_value']))
        # n_neg_regions = len(all_data['coopen_edges'][0]) if self.region_similarity else None
        enc_graph, open_value = all_data['enc_graph'].to(self.config.device), torch.tensor(all_data['open_value'])

        # weight = torch.tensor(all_data['weight'], device=self.config.device)



        model = VA_BRAID(n_layers = self.config.layers,
                     n_cells = self.n_cells,
                     n_regions = self.n_regions,
                     drop_out = self.config.drop_out,
                     feats_dim = self.config.feats_dim,
                     decoder = self.config.decoder
                     ).to(self.config.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)


        # start a new wandb run to track this script
        wandb.init(
        # set the wandb project where this run will be logged
        project="scATAC-DGL",
        # track hyperparameters and run metadata
        config=self.config)

        #######################   Start training model   #######################
        print(f"Start training ...")

        all_open_index, all_unopen_index = np.arange(len(open_value)), np.arange(len(all_data['unopen_edges'][0]))
        # all_uncoopen_index = np.arange(len(all_data['uncoexp_edges'][0])) if self.region_similarity else None

        if self.sample_rate == 1:
            pos_graph, pos_value = all_data['dec_graph'].to(self.config.device), all_data['open_value']
        # model.psedo_labeling(enc_graph)
        for iter_idx in range(self.config.iteration):
            # Sample un-expressed / un-co-expressed edges, construct negative graph
            neg_edges = {}

            unopen_sample_index = np.random.choice(all_unopen_index, n_neg_edges)
            neg_edges[('cell', 'open', 'region')] = (all_data['unopen_edges'][0][unopen_sample_index], all_data['unopen_edges'][1][unopen_sample_index])

            neg_graph = dgl.heterograph(neg_edges, num_nodes_dict={'cell': self.n_cells, 'region': self.n_regions})
            # add cell/region size factor to negative graph
            neg_graph.nodes['cell'].data['cs_factor'] = all_data['dec_graph'].nodes['cell'].data['cs_factor']
            neg_graph.nodes['region'].data['rs_factor'] = all_data['dec_graph'].nodes['region'].data['rs_factor']


            # Sample opened, construct positive graph
            if self.sample_rate != 1:
                pos_edges = {}

                open_sample_index = np.random.choice(all_open_index, n_pos_edges)
                pos_value = open_value[open_sample_index]
                # pos_weight = weight[open_sample_index]

                open_dec_edges = all_data['dec_graph'][('cell', 'access', 'region')].edges()
                pos_edges[('cell', 'access', 'region')] = (open_dec_edges[0][open_sample_index], open_dec_edges[1][open_sample_index])

                pos_graph = dgl.heterograph(pos_edges, num_nodes_dict={'cell': self.n_cells, 'region': self.n_regions})
                pos_graph.edata['count'] = pos_value
                # pos_graph.edata['weight'] = pos_weight

                # Add cell/gene size factor to positive graph
                pos_graph.nodes['cell'].data['cs_factor'] = all_data['dec_graph'].nodes['cell'].data['cs_factor']
                pos_graph.nodes['region'].data['rs_factor'] = all_data['dec_graph'].nodes['region'].data['rs_factor']


            # Feed forward
            # cell_2cluster = model.cluster_step(self.n_clusters)
            # pos_pre, neg_pre = model(enc_graph, pos_graph, neg_graph)
            # pos_pre, neg_pre, c_mu, c_logvar, r_mu, r_logvar = model.inference1(enc_graph, pos_graph, neg_graph)
            # pos_pre, neg_pre, c_mu, c_logvar, r_mu, r_logvar, cl_loss_node, cl_loss_cluster = model.inference2(enc_graph, pos_graph, neg_graph, cell_2cluster)

            # pos_pre, neg_pre, cl_loss_cluster, bpr_loss = model.inference4(enc_graph, pos_graph, neg_graph)

            # pos_pre, neg_pre, kl_loss, bpr_loss = model.inference5(enc_graph, pos_graph, neg_graph)

            # pos_pre, neg_pre, cl_loss_node = model.inference6(enc_graph, pos_graph, neg_graph)

            # bpr_loss, kl_loss, pos_pre, neg_pre = model.inference7(enc_graph, pos_graph, neg_graph)

            bpr_loss, kl_loss, NB_loss, reg_loss = model.inference(enc_graph, pos_graph.to(self.config.device), neg_graph.to(self.config.device))
            # bpr_loss, kl_loss, NB_loss, reg_loss, cl_loss_node, cl_loss_cluster = model.inference1(enc_graph, pos_graph, neg_graph, cell_2cluster=cell_2cluster)
            # bpr_loss, pos_pre, neg_pre = model.inference(enc_graph, pos_graph, neg_graph)


            # # 1. learning embeddings using gnn encoder
            # c_mu, c_logvar, r_mu, r_logvar, c_latent1, r_latent1, c_latent2, r_latent2 = model.inference(enc_graph)

            # # c_feat, r_feat = torch.split(embedding_view1, [self.n_cells, self.n_regions], dim=0)

            # 2. calculate poisson loss
            # pos_pre = model.decoder(pos_graph, c_latent1, r_latent1)
            # neg_pre = model.decoder(neg_graph, c_latent1, r_latent1)

            # Calculate loss for regularization
            # reg_loss = (model.cell_feature[pos_graph.edges()[0]].norm(2).pow(2) +
            #             model.region_feature[pos_graph.edges()[1]].norm(2).pow(2) +
            #             model.region_feature[neg_graph.edges()[1]].norm(2).pow(2)) / float(pos_graph.num_edges())
            
            # loss_access = -Poisson(pos_pre).log_prob(pos_value).mean()
            # loss_unaccess = -Poisson(neg_pre).log_prob(torch.zeros_like(neg_pre)).mean()
            # loss_access = NB_Loss(pos_value, pos_pre[0], pos_pre[1])
            # loss_unaccess = NB_Loss(torch.zeros_like(neg_pre[0]), neg_pre[0], neg_pre[1])

            # # loss_access = ZINB_Loss(pos_pre[0], pos_pre[1], pos_pre[2], pos_value)
            # # loss_unaccess = ZINB_Loss(neg_pre[0], neg_pre[1], neg_pre[2])

            # Calculate loss for regularization
            # reg_loss = (1 / 2) * (model.cell_feature.norm(2).pow(2) +
            #                   model.region_feature.norm(2).pow(2)) / float(self.n_cells + self.n_regions)

            # ridge = torch.square(pos_pre[0]).mean() + torch.square(neg_pre[0]).mean()
            # reg_loss = reg_loss + 1e-3 * ridge

            # 3. calculate kl loss
            # kl_loss = (model.kl_regularizer(c_mu, c_logvar) + model.kl_regularizer(r_mu, r_logvar)) / float(self.n_cells + self.n_regions)

            # # 4. calculate node-level contrastive loss
            # cl_loss_node = model.compute_cl_loss_node(c_latent1, r_latent1, c_latent2, r_latent2, pos_graph.nodes('cell'), pos_graph.nodes('region'))

            # # 5. calculate cluster-level contrasive loss
            # cl_loss_cluster = model.compute_cl_loss_cluster(c_latent1, c_latent2, pos_graph.nodes('cell'), user_2cluster)

            # loss = loss_access + loss_unaccess + 1e-3 * reg_loss + kl_loss + 0*cl_loss_node + 0*cl_loss_cluster
            # loss = loss_access + loss_unaccess + 1e-4 * reg_loss

            # poisson_loss, cl_loss = model.calculate_all_loss(enc_graph, pos_graph, neg_graph, pos_value, cell_2cluster)

            # loss = (loss_access + loss_unaccess + 1e-3 * reg_loss) + cl_loss_node*0.
            loss = bpr_loss + kl_loss + self.config.beta * reg_loss + NB_loss * self.config.alpha
            # loss = bpr_loss*self.config.alpha + 1e-3 * reg_loss + loss_access + loss_unaccess
            # loss = bpr_loss + kl_loss + 1e-3 * reg_loss


            # if self.verbose:
            #     count_loss_access += loss_access.item()
            #     count_loss_unaccess += loss_unaccess.item()

            all_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if iter_idx % 10 == 0:
            #     model.eval()
            #     with torch.no_grad():
            #         model.psedo_labeling(enc_graph)

            wandb.log({"bpr_loss": (bpr_loss).item(), "kl_loss": kl_loss,
                       "NB_loss": (NB_loss).item(), "loss": loss.item()})

            if self.verbose and self.cell_type is not None and (iter_idx + 1) % (self.log_interval) == 0:
                model.eval()
                with torch.no_grad():
                    c_feat, r_feat, c_last, r_last = model.encode(enc_graph)
                    c_feat = model.reparameter(c_feat, c_last)
                    # c_feat, r_feat, c_last, _,_,_ = model.encode2(enc_graph)
                    # c_feat, r_feat, c_last, _,_,_,_,_ = model.encode2(enc_graph)
                    # c_latent1, peak_view1 = torch.split(embedding_view1, [model.n_cells, model.n_regions], dim=0)
                # model.train()
                    # Cell embeddings
                start_time = time.time()
                adata.obsm['e0'] = model.cell_feature.detach().cpu().numpy()  # Return initial cell embedding
                adata.obsm['e2'] = c_last.cpu().numpy()  # Return the final layer of cell embedding
                adata.obsm['feat'] = c_feat.cpu().numpy()  # Return the weighted cell embeddings

                # # louvain
                adata = getNClusters(adata, use_rep=self.use_rep, n_cluster=self.n_clusters, method='leiden')
                y_pred_l = np.array(adata.obs['leiden'])

                nmi_l, ari_l = calculate_metric(self.cell_type, y_pred_l)
                end_time = time.time()
                print('%04d Loss=%.2f, ARI= %.4f, NMI= %.4f, time= %.2f' % (iter_idx + 1, loss.item(), ari_l, nmi_l, (end_time - start_time)))
                # print('%04d Loss=%.2f, ARI= %.4f, NMI= %.4f' % (iter_idx + 1, loss.item(), ari_l, nmi_l))

                # Cell embeddings
                # adata.obsm['e0'] = model.cell_feature.detach().cpu().numpy()  # Return initial cell embedding
                # adata.obsm['e2'] = c_last.cpu().numpy()  # Return the final layer of cell embedding
                # adata.obsm['feat'] = c_feat.cpu().numpy()  # Return the weighted cell embeddings

                # # kmeans
                # # adata = kmeans(adata, self.n_clusters, use_rep=self.use_rep)
                # # y_pred_k = np.array(adata.obs['kmeans'])

                # # louvain
                # adata = getNClusters(adata, use_rep=self.use_rep, n_cluster=self.n_clusters)
                # y_pred_l = np.array(adata.obs['louvain'])

                # # nmi_k, ari_k = calculate_metric(self.cell_type, y_pred_k)
                # # print('Clustering Kmeans %d: NMI= %.4f, ARI= %.4f' % (iter_idx + 1, nmi_k, ari_k))

                # nmi_l, ari_l = calculate_metric(self.cell_type, y_pred_l)
                # print('%04d Loss=%.2f, ARI= %.4f, NMI= %.4f' % (iter_idx + 1, loss.item(), ari_l, nmi_l))

                wandb.log({"ARI": ari_l, "NMI": nmi_l})

                all_ari_l.append(ari_l)
                all_nmi_l.append(nmi_l)


                if ari_l > best_ari_l:
                    best_ari_l = ari_l
                    best_nmi_l = nmi_l
                    best_iter_l = iter_idx + 1
        ## End of training
        model.eval()
        with torch.no_grad():
            c_feat, g_feat, c_last, g_last = model.encode(enc_graph)
            c_feat = model.reparameter(c_feat, c_last)
            g_feat = model.reparameter(g_feat, g_last)
            # c_feat, g_feat, c_last, _, _, _ = model.encode1(enc_graph)

        adata.obsm['e0'] = model.cell_feature.data.cpu().numpy()  # Return initial cell embedding
        adata.obsm['e2'] = c_last.cpu().numpy()  # Return the final layer of cell embedding
        adata.obsm['feat'] = c_feat.cpu().numpy()  # Return the weighted cell embeddings
        adata.varm['feat'] = g_feat.cpu().numpy()  # Return the final layer's region embeddings

        if self.verbose and self.cell_type is not None:
            print(
                f'[END] For Leiden, Best Iter : {best_iter_l} Best ARI : {best_ari_l:.4f}, Best NMI : {best_nmi_l:.4f}')

        wandb.finish()
        adata.write('/home/suyanchi/project/atac/results/clustering/our/' + self.config.dataset + '.h5ad')
        #######################   Impute expression matrix (Optional) ########################
        if self.impute:
            all_open_cell, all_open_region = np.meshgrid(np.arange(self.n_cells), np.arange(self.n_regions))
            all_open_cell, all_open_region = all_open_cell.reshape(-1), all_open_region.reshape(-1)

            all_dec_graph = dgl.heterograph({('cell', 'open', 'region'): (all_open_cell, all_open_region)},
                                            num_nodes_dict={'cell': self.n_cells, 'region': self.n_regions}).to(self.config.device)
            all_dec_graph.nodes['cell'].data['cs_factor'] = all_data['dec_graph'].nodes['cell'].data['cs_factor'].to(self.config.device)
            all_dec_graph.nodes['region'].data['rs_factor'] = all_data['dec_graph'].nodes['region'].data['rs_factor'].to(self.config.device)

            model.eval()
            with torch.no_grad():
                all_exp = model(enc_graph, all_dec_graph)

            all_exp = all_exp.data.cpu().numpy().reshape(self.n_regions, self.n_cells).T

            adata.obsm['imputed'] = all_exp

        if self.save_model:
            torch.save(model, 'trained_model/'+ self.config.dataset + '.pt')

        del model
        # del all_exp
        gc.collect()
        torch.cuda.empty_cache()

        return adata