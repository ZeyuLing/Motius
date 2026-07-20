import torch
import torch.nn as nn
from .base import BaseLosses


class CommitLoss(nn.Module):
    """
    Useless Wrapper
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, commit, commit2, **kwargs):
        return commit


class MotLosses(BaseLosses):
    
    def __init__(self, cfg, stage, num_joints, **kwargs):
        # Save parameters
        self.stage = stage
        recons_loss = cfg.LOSS.ABLATION.RECONS_LOSS
        self.predict_epsilon = cfg.ABLATION.PREDICT_EPSILON

        # Define losses
        losses = []
        params = {}
        if stage == "vae":
            losses.append("recons_feature")
            params['recons_feature'] = cfg.LOSS.LAMBDA_FEATURE

            losses.append("recons_velocity")
            params['recons_velocity'] = cfg.LOSS.LAMBDA_VELOCITY

            # KL loss
            losses.append("kl_motion")
            params['kl_motion'] = cfg.LOSS.LAMBDA_KL

            # losses.append("vq_commit")
            # params['vq_commit'] = cfg.LOSS.LAMBDA_COMMIT
        elif stage in ["lm_pretrain", "lm_instruct", "lm_finetune", "lm_t2m", 'lm_adaptor_pretrain', 'lm_fixdec']:
            losses.append("gpt_loss")
            params['gpt_loss'] = cfg.LOSS.LAMBDA_CLS
            losses.append("diff_loss")
            params['diff_loss'] = cfg.LOSS.LAMBDA_DIFF
            # losses.append("boundary_loss")
            # params['boundary_loss'] = cfg.LOSS.LAMBDA_BOUND

            # losses.append("inst_loss") 
            # params[loss] = 1
            # losses.append("x_loss")
            # params[loss] = 1
            # if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
            #     # prior noise loss
            #     losses.append("prior_loss")
            #     params[loss] = cfg.LOSS.LAMBDA_PRIOR

            # if stage in ["lm_pretrain"]:
            #     losses.append("recons_feature")
            #     params['recons_feature'] = cfg.LOSS.LAMBDA_FEATURE

            #     losses.append("recons_velocity")
            #     params['recons_velocity'] = cfg.LOSS.LAMBDA_VELOCITY

        # Define loss functions & weights
        losses_func = {}
        for loss in losses:
            if loss.split('_')[0] == 'recons':
                if recons_loss == "l1":
                    losses_func[loss] = nn.L1Loss
                elif recons_loss == "l2":
                    losses_func[loss] = nn.MSELoss
                elif recons_loss == "l1_smooth":
                    losses_func[loss] = nn.SmoothL1Loss
            # KL loss
            elif loss.split('_')[0] == 'kl':
                losses_func[loss] = KLLoss()
            elif loss.split('_')[0] == ['inst', 'x']:
                losses_func[loss] = nn.MSELoss(reduction='mean')
            elif loss.split('_')[1] in [
                    'commit', 'loss', 'gpt', 'm2t2m', 't2m2t', 'diff'
            ]:
                losses_func[loss] = CommitLoss
            elif loss.split('_')[1] in ['cls', 'lm']:
                losses_func[loss] = nn.CrossEntropyLoss
            else:
                raise NotImplementedError(f"Loss {loss} not implemented.")

        super().__init__(cfg, losses, params, losses_func, num_joints,
                         **kwargs)

    # def forward_loss(self, z, target):
    #     bsz, seq_len, _ = target.shape
    #     target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
    #     z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
    #     loss = self.diffloss(z=z, target=target)
    #     return loss
    
    def update(self, rs_set, gpt_coef=1.):
        '''Update the losses'''
        total: float = 0.0

        if self.stage in ["vae"]:
            total += self._update_loss("recons_feature", rs_set['m_rst'],
                                       rs_set['m_ref'])
            # total += self._update_loss("recons_joints", rs_set['joints_rst'], rs_set['joints_ref'])
            total += self._update_loss("kl_motion", rs_set['dist_m'], rs_set['dist_ref'])

            nfeats = rs_set['m_rst'].shape[-1]
            if nfeats in [263, 135 + 263, 313, 623, 322]:
                if nfeats == 135 + 263:
                    vel_start = 135 + 4
                    vel_end = (self.num_joints - 1) * 3 +vel_start
                elif nfeats == 263:
                    vel_start = 4
                    vel_end = (self.num_joints - 1) * 3 +vel_start
                elif nfeats == 313:
                    vel_start = 4
                    vel_end = (self.num_joints - 1) * 3 +vel_start
                    # vel_start = (self.num_joints - 1) * 3
                    # vel_end = self.num_joints * 3 +vel_start
                elif nfeats == 623:
                    vel_start = 4
                    vel_end = (self.num_joints - 1) * 3 +vel_start
                    # vel_start = (self.num_joints - 1) * 9
                    # vel_end = self.num_joints * 3 +vel_start
                elif nfeats == 322:
                    vel_start = 3
                    vel_end = (self.num_joints - 1) * 3 +vel_start
                total += self._update_loss(
                    "recons_velocity",
                    rs_set['m_rst'][..., vel_start:vel_end],
                    rs_set['m_ref'][..., vel_start:vel_end])
            else:
                if self._params['recons_velocity'] != 0.0:
                    raise NotImplementedError(
                        "Velocity not implemented for nfeats = {})".format(nfeats))
            # total += self._update_loss("vq_commit", rs_set['loss_commit'],
            #                            rs_set['loss_commit'])


        if self.stage in ["lm_pretrain", "lm_instruct", "lm_finetune", "lm_t2m", 'lm_adaptor_pretrain', 'lm_fixdec']:
            total += self._update_loss("gpt_loss", rs_set['outputs'].loss,
                                       rs_set['outputs'].loss, coef=gpt_coef)
            total += self._update_loss("diff_loss", rs_set['outputs'].diff_loss, 
                                       rs_set['outputs'].diff_loss)
            # self._update_loss('boundary_loss', rs_set['outputs'].boundary_loss, 
            #                     rs_set['outputs'].boundary_loss)
            # if self.predict_epsilon:
            #     total += self._update_loss("inst_loss", rs_set['noise_pred'],
            #                                rs_set['noise'])
            # # predict x
            # else:
            #     total += self._update_loss("x_loss", rs_set['pred'],
            #                                rs_set['latent'])
            # if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
            #     # loss - prior loss
            #     total += self._update_loss("prior_loss", rs_set['noise_prior'],
            #                                rs_set['dist_m1'])
            # total += self.forward_loss(z=rs_set['hidden'], target=rs_set['tokens_ref'], mask=None)

        # Update the total loss
        self.total += total.detach()
        self.count += 1

        return total

class KLLoss:

    def __init__(self):
        pass

    def __call__(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"