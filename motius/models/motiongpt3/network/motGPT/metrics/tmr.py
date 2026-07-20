from typing import List
import os
import torch
from torch import Tensor
from torchmetrics import Metric
# from torchmetrics.functional import pairwise_euclidean_distance
from .utils import *
from motius.models.motiongpt3.network.motGPT.config import instantiate_from_config
from motius.models.motiongpt3.network.motGPT.metrics.tmr_metrics import all_contrastive_metrics
from motius.models.motiongpt3.network.motGPT.metrics.tmr_utils import default_collate, collate_x_dict, length_to_mask
import pytorch_lightning as pl


# x.T will be deprecated in pytorch
def transpose(x):
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))

def get_sim_matrix(x, y):
    x_logits = torch.nn.functional.normalize(x, dim=-1)
    y_logits = torch.nn.functional.normalize(y, dim=-1)
    sim_matrix = x_logits @ transpose(y_logits)
    return sim_matrix


class TMRMetrics(Metric):
    def __init__(self,
                 cfg,
                 dataname='humanml3d',
                #  top_k=3,
                 R_size=32,
                 diversity_times=300,
                 dist_sync_on_step=True,
                 threshold_selfsim_metrics=0.95,
                 protocol='normal',  # 'normal', 'threshold', 'nsim', 'guo'
                 fact=1.0,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg
        self.dataname = dataname
        self.name = "t2m metrics"
        self.fact = fact
        self.vae = True
        self.threshold_selfsim_metrics = threshold_selfsim_metrics if protocol=='threshold' else None
        self.protocol = protocol
        self.R_size = R_size
        # self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

        # # self.top_k = top_k
        # # self.text = 'lm' in cfg.TRAIN.STAGE and cfg.model.params.task == 't2m'
        # self.diversity_times = diversity_times

        # Chached batches
        self.add_state("rec_m_latents", default=[], dist_reduce_fx=None)
        self.add_state("gt_m_latents", default=[], dist_reduce_fx=None)
        self.add_state("t_latents", default=[], dist_reduce_fx=None)
        self.add_state("text_sent_emb", default=[], dist_reduce_fx=None)

        # T2M Evaluator
        self._get_tmr_evaluator(cfg)

    def _get_tmr_evaluator(self, cfg):
        """
        load TMR text encoder and motion encoder for evaluating
        """
        # init module
        # self.tmr_model = model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)
        # self.tmr_model = instantiate_from_config(cfg.METRIC.TMR.tmr_motionencoder)
        self.tmr_motionencoder = instantiate_from_config(cfg.METRIC.TMR.tmr_motionencoder)
        # self.tmr_motiondecoder = instantiate_from_config(cfg.METRIC.TMR.tmr_motiondecoder)
        self.tmr_textencoder = instantiate_from_config(cfg.METRIC.TMR.tmr_textencoder)

        self.text_to_token_emb = instantiate_from_config(cfg.METRIC.TMR.text_to_token_emb)
        self.text_to_sent_emb = instantiate_from_config(cfg.METRIC.TMR.text_to_sent_emb)

        # load pretrianed
        # # self.tmr_model.motion_encoder.load_state_dict(os.path.join(cfg.METRIC.TMR.tmr_path,'last_weights','motion_encoder.pt'))
        # # self.tmr_model.motion_decoder.load_state_dict(os.path.join(cfg.METRIC.TMR.tmr_path,'last_weights','motion_decoder.pt'))
        # # self.tmr_model.text_encoder.load_state_dict(os.path.join(cfg.METRIC.TMR.tmr_path,'last_weights','text_encoder.pt'))
        self.tmr_motionencoder.load_state_dict(torch.load(os.path.join(cfg.METRIC.TMR.tmr_path,'last_weights','motion_encoder.pt')))
        self.tmr_textencoder.load_state_dict(torch.load(os.path.join(cfg.METRIC.TMR.tmr_path,'last_weights','text_encoder.pt')))

        # freeze params
        # self.tmr_model.eval()
        # for p in self.tmr_model.parameters():
        #     p.requires_grad = False
        self.tmr_motionencoder.eval()
        for p in self.tmr_motionencoder.parameters():
            p.requires_grad = False
        self.tmr_textencoder.eval()
        for p in self.tmr_textencoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def compute(self, sanity_flag):
        # pl.seed_everything(1234)
        # torch.random.seed_everything(1234)
        print(' in tmr metrics computing')
        # count = self.count.item()
        count_seq = self.count_seq.item()
        # Init metrics dict
        # metrics = {metric: getattr(self, metric) for metric in self.metrics}
        metrics = {}

        # Jump in sanity check stage
        if sanity_flag:
            return metrics

        # Cat cached batches and shuffle
        shuffle_idx = torch.randperm(count_seq)

        all_genmotions = torch.cat(self.rec_m_latents, axis=0).cpu()[shuffle_idx, :]
        all_gtmotions = torch.cat(self.gt_m_latents,axis=0).cpu()[shuffle_idx, :]

        all_texts = torch.cat(self.t_latents, axis=0).cpu()[shuffle_idx, :]
        sent_emb = torch.cat(self.text_sent_emb,axis=0).cpu()[shuffle_idx, :]

        all_gen_metrics = []
        all_gt_metrics = []
        gen_sim_score = []
        gt_sim_score = []
        if self.protocol == 'guo':
            R_size = self.R_size
        else:
            R_size = count_seq
        assert count_seq >= R_size
        # gen_sim_matrix = get_sim_matrix(group_texts, all_genmotions).cpu().numpy()
        # gt_sim_matrix = get_sim_matrix(all_texts, all_gtmotions).cpu().numpy()
        for i in range(count_seq // R_size):
            group_texts = all_texts[i*R_size: (i+1)*R_size]
            group_gen_motions = all_genmotions[i*R_size: (i+1)*R_size]
            group_gt_motions = all_gtmotions[i*R_size: (i+1)*R_size]
            group_sent_emb = sent_emb[i*R_size: (i+1)*R_size]

            gen_sim_matrix = get_sim_matrix(group_texts, group_gen_motions).cpu().numpy()
            gen_sim_score.append(np.diag(gen_sim_matrix))
            gen_contrastive_metrics = all_contrastive_metrics(
                gen_sim_matrix,
                emb=group_sent_emb.cpu().numpy(),
                threshold=self.threshold_selfsim_metrics,
            )
            all_gen_metrics.append(gen_contrastive_metrics)
            # metrics['gen_metrics'].update(gen_contrastive_metrics)

            gt_sim_matrix = get_sim_matrix(group_texts, group_gt_motions).cpu().numpy()
            gt_sim_score.append(np.diag(gt_sim_matrix))
            gt_contrastive_metrics = all_contrastive_metrics(
                gt_sim_matrix,
                emb=group_sent_emb.cpu().numpy(),
                threshold=self.threshold_selfsim_metrics,
            )
            all_gt_metrics.append(gt_contrastive_metrics)
            # metrics['gt_metrics'].update(gt_contrastive_metrics)

        # metrics['gen_metrics'] = {}
        for key in all_gen_metrics[0].keys():
            metrics['gen_'+key] = round(
                float(np.mean([metrics[key] for metrics in all_gen_metrics])), 2
            )
        # metrics['gt_metrics'] = {}
        for key in all_gen_metrics[0].keys():
            metrics['gt_'+key] = round(
                float(np.mean([metrics[key] for metrics in all_gt_metrics])), 2
            )
        
        R_count = count_seq // R_size * R_size
        metrics['gen_sim_score'] = np.concatenate(gen_sim_score).mean()
        metrics['gt_sim_score'] = np.concatenate(gt_sim_score).mean()
        # tensor -> numpy for FID
        all_genmotions = all_genmotions.numpy()
        all_gtmotions = all_gtmotions.numpy()
        
        # # Compute fid
        # mu, cov = calculate_activation_statistics_np(all_genmotions)
        # gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions)
        # metrics["FID"] = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)

        # # Compute diversity
        # assert count_seq > self.diversity_times
        # metrics["Diversity"] = calculate_diversity_np(all_genmotions,
        #                                               self.diversity_times)
        # metrics["gt_Diversity"] = calculate_diversity_np(
        #     all_gtmotions, self.diversity_times)

        # Reset
        self.reset()

        return {**metrics}

    @torch.no_grad()
    def update(self,
               feats_ref: Tensor,
               feats_rst: Tensor,
               lengths_ref: List[int],
               lengths_rst: List[int],
               texts: List[str]):

        # self.count += sum(lengths_ref)
        self.count_seq += len(lengths_ref)

        nan_mask = False
        flag = False
        # T2m motion encoder
        # align_idx = np.argsort(lengths_ref)[::-1].copy()
        # feats_ref = feats_ref[align_idx]
        lengths_ref = np.array(lengths_ref)#[align_idx]

        gtmotion_embeddings = self.get_motion_embeddings(
            feats_ref, lengths_ref)
        nan_mask = nan_mask | (torch.isnan(gtmotion_embeddings).sum(-1)>0)
        if torch.isnan(gtmotion_embeddings).sum() > 0:
            flag = True
            print('nan in gtmotion_embeddings', torch.isnan(gtmotion_embeddings).sum(-1))
        # cache = [0] * len(lengths_ref)
        # for i in range(len(lengths_ref)):
        #     cache[align_idx[i]] = gtmotion_embeddings[i:i + 1]
        # self.gt_m_latents.extend(cache)

        # align_idx = np.argsort(lengths_rst)[::-1].copy()
        # feats_rst = feats_rst[align_idx]
        lengths_rst = np.array(lengths_rst)#[align_idx]
        recmotion_embeddings = self.get_motion_embeddings(
            feats_rst, lengths_rst)
        nan_mask = nan_mask | (torch.isnan(recmotion_embeddings).sum(-1)>0)
        if torch.isnan(recmotion_embeddings).sum() > 0:
            flag = True
            print('nan in recmotion_embeddings', torch.isnan(recmotion_embeddings).sum(-1))
        # cache = [0] * len(lengths_rst)
        # for i in range(len(lengths_rst)):
        #     cache[align_idx[i]] = recmotion_embeddings[i:i + 1]
        # self.rec_m_latents.extend(cache)

        # T2m text encoder
        text_emb = self.get_text_embeddings(texts, device=recmotion_embeddings.device)
        nan_mask = nan_mask | (torch.isnan(text_emb).sum(-1)>0)
        if torch.isnan(text_emb).sum() > 0:
            flag = True
            print('nan in text_emb', torch.isnan(text_emb).sum(-1))

        self.gt_m_latents.append(gtmotion_embeddings[~nan_mask])
        self.rec_m_latents.append(recmotion_embeddings[~nan_mask])
        text_embeddings = torch.flatten(text_emb, start_dim=1).detach()
        self.t_latents.append(text_embeddings[~nan_mask])
        sent_emb = self.text_to_sent_emb(texts)
        sent_emb = default_collate(sent_emb)
        # print(nan_mask.shape, sent_emb.shape)
        if flag:
            print(sent_emb.shape, nan_mask.shape)
        self.text_sent_emb.append(sent_emb[(~nan_mask).cpu()])
        self.count_seq -= nan_mask.sum()

    def process_encoded_into_latent(self, encoded, sample_mean=True):
        if self.vae:
            dists = encoded.unbind(1)
            mu, logvar = dists
            if sample_mean:
                latent_vectors = mu
            else:
                # Reparameterization trick
                std = logvar.exp().pow(0.5)
                eps = std.data.new(std.size()).normal_()
                latent_vectors = mu + self.fact * eps * std
        else:
            dists = None
            (latent_vectors,) = encoded.unbind(1)
        return latent_vectors.detach()
        
    def get_motion_embeddings(self, feats: Tensor, lengths: List[int], sample_mean=True):
        mask = length_to_mask(lengths, device=feats.device)
        motion_x_dict = {'x': feats, "length": lengths, "mask": mask}
        # motion_x_dict = collate_x_dict(motion_x_dict)
        encoded = self.tmr_motionencoder(motion_x_dict)
        latent_vectors = self.process_encoded_into_latent(encoded, sample_mean=sample_mean)
        # if self.vae:
        #     dists = encoded.unbind(1)
        #     mu, logvar = dists
        #     if sample_mean:
        #         latent_vectors = mu
        #     else:
        #         # Reparameterization trick
        #         std = logvar.exp().pow(0.5)
        #         eps = std.data.new(std.size()).normal_()
        #         latent_vectors = mu + self.fact * eps * std
        # else:
        #     dists = None
        #     (latent_vectors,) = encoded.unbind(1)
        return latent_vectors

    def get_text_embeddings(self, texts, device, sample_mean=True):
        text_x_dict = self.text_to_token_emb(texts)
        text_x_dict = collate_x_dict(text_x_dict, device=device)
        encoded = self.tmr_textencoder(text_x_dict)
        latent_vectors = self.process_encoded_into_latent(encoded, sample_mean=sample_mean)
        # if self.vae:
        #     dists = encoded.unbind(1)
        #     mu, logvar = dists
        #     if sample_mean:
        #         latent_vectors = mu
        #     else:
        #         # Reparameterization trick
        #         std = logvar.exp().pow(0.5)
        #         eps = std.data.new(std.size()).normal_()
        #         latent_vectors = mu + self.fact * eps * std
        # else:
        #     dists = None
        #     (latent_vectors,) = encoded.unbind(1)
        return latent_vectors