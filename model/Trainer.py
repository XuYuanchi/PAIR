import gc
import torch
import torch.nn as nn
from contextlib import nullcontext
import dgl

import numpy as np
import wandb
import os

from sklearn import preprocessing

from utils import make_graph, calculate_metric, getNClusters
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
            # impute: bool = False,
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
        self.impute = config.impute
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
            with open('/home/suyanchi/project/atac/temp/' + self.config.dataset + '.pkll1', 'rb') as f:
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

        best_ari_k, best_ari_l = 0, 0
        best_nmi_k, best_nmi_l = 0, 0
        all_ari_k, all_ari_l = [], []
        all_nmi_k, all_nmi_l = [], []

        best_iter_k, best_iter_l = -1, -1

        #######################   train/test data   #######################

        n_pos_edges, n_neg_edges = int(self.sample_rate * len(all_data['open_value'])), int(self.sample_rate * len(all_data['open_value']))
        # n_neg_regions = len(all_data['coopen_edges'][0]) if self.region_similarity else None
        enc_graph, open_value = all_data['enc_graph'].to(self.config.device), torch.tensor(all_data['open_value'])




        model = VA_BRAID(n_layers = self.config.layers,
                     n_cells = self.n_cells,
                     n_regions = self.n_regions,
                     drop_out = self.config.drop_out,
                     feats_dim = self.config.feats_dim,
                     decoder = self.config.decoder,
                     use_cell2cell = self.config.use_cell2cell
                     ).to(self.config.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        # scaler = torch.cuda.amp.GradScaler()

        os.environ["WANDB_MODE"] = "offline"
        # start a new wandb run to track this script
        wandb.init(
        # set the wandb project where this run will be logged
        project="atac",
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


            bpr_loss, kl_loss, NB_loss, reg_loss = model.inference(enc_graph, pos_graph.to(self.config.device), neg_graph.to(self.config.device))

            loss = bpr_loss + kl_loss + self.config.beta * reg_loss + NB_loss * self.config.alpha
 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


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
                # adata.obsm['e2'] = c_last.cpu().numpy()  # Return the final layer of cell embedding
                adata.obsm['feat'] = c_feat.cpu().numpy()  # Return the weighted cell embeddings

                # # louvain
                adata = getNClusters(adata, use_rep=self.use_rep, n_cluster=self.n_clusters, method='leiden')
                y_pred_l = np.array(adata.obs['leiden'])

                nmi_l, ari_l = calculate_metric(self.cell_type, y_pred_l)
                end_time = time.time()
                print('%04d Loss=%.2f, ARI= %.4f, NMI= %.4f, time= %.2f' % (iter_idx + 1, loss.item(), ari_l, nmi_l, (end_time - start_time)))



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
            c_feat, r_feat, c_last, r_last = model.encode(enc_graph)
            c_feat = model.reparameter(c_feat, c_last)
            r_feat = model.reparameter(r_feat, r_last)
            rs_factor = model.region_factors
            # c_feat, g_feat, c_last, _, _, _ = model.encode1(enc_graph)

        adata.obsm['e0'] = model.cell_feature.data.cpu().numpy()  # Return initial cell embedding
        # adata.obsm['e2'] = c_last.cpu().numpy()  # Return the final layer of cell embedding
        adata.obsm['feat'] = c_feat.cpu().numpy()  # Return the weighted cell embeddings
        adata.varm['feat'] = r_feat.cpu().numpy()  # Return the final layer's region embeddings
        adata.var['region_factors'] = rs_factor.detach().cpu().numpy()

        if self.verbose and self.cell_type is not None:
            print(
                f'[END] For Leiden, Best Iter : {best_iter_l} Best ARI : {best_ari_l:.4f}, Best NMI : {best_nmi_l:.4f}')

        wandb.finish()

        def decode_forward1_in_blocks(
                model,
                enc_graph,
                cs_factor,         # shape: (n_cells,)   (from all_data['dec_graph'].nodes['cell'].data['cs_factor'])
                rs_factor,         # shape: (n_regions,) (通常是 model.region_factors 或 region节点上的因子)
                n_cells,
                n_regions,
                device,
                cell_bs=None,      # 每次处理多少个cell；默认一次性处理所有cell
                region_bs=4096,    # 每次处理多少个region；按显存调
                use_amp=True,      # 开启混合精度省显存
            ):
            model.eval()

            # 1) 编码一次
            with torch.no_grad():
                c_feat, r_feat, c_last, r_last = model.encode(enc_graph)
                c_feat = model.reparameter(c_feat, c_last).to(device, non_blocking=True).float()   # (Nc, k)
                r_feat = model.reparameter(r_feat, r_last).to(device, non_blocking=True).float()   # (Nr, k)

            cs = cs_factor.to(device, non_blocking=True).view(-1)          # (Nc,)
            rs = rs_factor.to(device, non_blocking=True).view(-1)          # (Nr,)

            # 2) 预分配CPU结果（可改为 np.memmap 进一步省内存）
            mu_out   = np.empty((n_cells, n_regions), dtype=np.float32)
            disp_out = np.empty((n_cells, n_regions), dtype=np.float32)
            score_out= np.empty((n_cells, n_regions), dtype=np.float32)

            # cell 方向分块（大多数场景 Nc 比较小，可以一次性处理）
            if cell_bs is None:
                cell_bs = n_cells

            # 3) 双层分块
            def normalize_device(dev):
                # dev 可以是 "cuda:1" / "cuda:0" / "cpu" / torch.device
                dev = torch.device(dev) if not isinstance(dev, torch.device) else dev
                if dev.type == 'cuda':
                    if not torch.cuda.is_available():
                        return torch.device('cpu')
                    idx = 0 if dev.index is None else dev.index
                    if idx >= torch.cuda.device_count():
                        # 回落到 cuda:0；如果你想直接回落到 CPU，把下一行改成 return torch.device('cpu')
                        return torch.device('cuda:0')
                return dev

            device = normalize_device(device)  # 你传进来的 self.config.device

            # 只在有效 CUDA 上启用 autocast；优先 bf16（Ampere+），否则 fp16；CPU 上禁用
            if device.type == 'cuda' and torch.cuda.is_available():
                try:
                    with torch.cuda.device(device):
                        major = torch.cuda.get_device_properties(device).major
                    amp_dtype = torch.bfloat16 if major >= 8 else torch.float16  # Ampere(8.x)+ 支持 bf16
                except Exception:
                    amp_dtype = torch.float16
                amp_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)
            else:
                amp_ctx = nullcontext()
            with torch.no_grad():
                for i0 in range(0, n_cells, cell_bs):
                    i1 = min(i0 + cell_bs, n_cells)
                    C = c_feat[i0:i1]                        # (Bc, k)
                    cs_blk = cs[i0:i1]                       # (Bc,)

                    for j0 in range(0, n_regions, region_bs):
                        j1 = min(j0 + region_bs, n_regions)
                        R = r_feat[j0:j1]                    # (Br, k)
                        rs_blk = rs[j0:j1]                   # (Br,)

                        with amp_ctx:
                            # score = u_dot_v('h','h')
                            # (Bc, k) @ (k, Br) -> (Bc, Br)
                            S = torch.matmul(C, R.T)

                            # h_d = u_mul_v('h','h')  -> 逐元素乘
                            # 先广播得到 (Bc, Br, k)，再展平为 (Bc*Br, k) 喂给 MLP
                            # 注意：这一步是内存热点，控制好 Bc*Br 的大小
                            HD = (C[:, None, :] * R[None, :, :]).reshape(-1, C.shape[1])  # (Bc*Br, k)

                            mu_raw   = model.decoder.dec_mean(HD)  # (Bc*Br, 1) or (Bc*Br,)
                            disp_raw = model.decoder.dec_disp(HD)  # 同上

                            # reshape 回 (Bc, Br) 并施加 rs/cs + 激活，保持与 forward1 一致
                            mu_raw   = mu_raw.view(i1 - i0, j1 - j0)
                            disp_raw = disp_raw.view(i1 - i0, j1 - j0)

                            mu_blk   = model.decoder.dec_mean_act( mu_raw * rs_blk[None, :] ) * cs_blk[:, None]
                            disp_blk = model.decoder.dec_disp_act( disp_raw * rs_blk[None, :] )

                        # 回写到CPU numpy
                        mu_out[i0:i1,   j0:j1] = mu_blk.detach().float().cpu().numpy()
                        disp_out[i0:i1, j0:j1] = disp_blk.detach().float().cpu().numpy()
                        score_out[i0:i1,j0:j1] = S.detach().float().cpu().numpy()

                        # 释放中间量
                        del R, rs_blk, S, HD, mu_raw, disp_raw, mu_blk, disp_blk
                        torch.cuda.empty_cache()

                    del C, cs_blk
                    torch.cuda.empty_cache()

            return mu_out, disp_out, score_out


        if self.impute:
            device = self.config.device
            cs_factor = all_data['dec_graph'].nodes['cell'].data['cs_factor'].float()
            # 你之前传的是 model.region_factors；也可以用 dec_graph 上的
            rs_factor = model.region_factors.float()  # 或 all_data['dec_graph'].nodes['region'].data['rs_factor'].float()

            mu, _, score = decode_forward1_in_blocks(
                model=model,
                enc_graph=enc_graph,
                cs_factor=cs_factor,
                rs_factor=rs_factor,
                n_cells=self.n_cells,
                n_regions=self.n_regions,
                device=device,
                cell_bs=None,       # 若内存紧张可设为比如 1024
                region_bs=4096,     # 视显存调大/调小
                use_amp=True
            )

            adata.layers['impute_count'] = mu         # 对应 forward1 返回的 mu
            adata.layers['impute']       = score      # 你原先 b=all_exp[-1] 是 score
            # 如需保存 disp：
            # adata.layers['impute_disp']  = disp

        if self.save_model:
            torch.save(model, 'trained_model/'+ self.config.dataset + '.pt')

        del model
        # del all_exp
        gc.collect()
        torch.cuda.empty_cache()

        return adata