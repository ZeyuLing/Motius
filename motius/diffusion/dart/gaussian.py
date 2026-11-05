# Sampling mathematics derived from DART and OpenAI guided-diffusion.
import enum, math
from copy import deepcopy
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
from .math import mean_flat, normal_kl, discretized_gaussian_log_likelihood

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, scale_betas=1.0):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == 'linear':
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == 'cosine':
        return betas_for_alpha_bar(num_diffusion_timesteps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)
    else:
        raise NotImplementedError(f'unknown beta schedule: {schedule_name}')

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """
    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()

class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()

class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL

class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(self, *, betas, model_mean_type, model_var_type, loss_type, rescale_timesteps=False, lambda_rcxyz=0.0, lambda_vel=0.0, lambda_pose=1.0, lambda_orient=1.0, lambda_loc=1.0, data_rep='rot6d', lambda_root_vel=0.0, lambda_vel_rcxyz=0.0, lambda_fc=0.0, lambda_smpl_joints=0.0, lambda_smpl_vertices=0.0, lambda_joints_consistency=0.0, lambda_joints_delta=0.0, lambda_transl_delta=0.0, lambda_orient_delta=0.0, lambda_first_joints_delta=0.0, lambda_first_transl_delta=0.0, lambda_first_orient_delta=0.0, lambda_rel_joints_delta=0.0, lambda_rel_transl_delta=0.0, lambda_rel_orient_delta=0.0, lambda_delta_norm=0.0, lambda_smooth=0.0, lambda_dct=0.0, lambda_jerk=0.0, ignore_history=False, mask_thresh_step=1.0):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.ignore_history = ignore_history
        self.mask_thresh_step = mask_thresh_step
        self.data_rep = data_rep
        if data_rep != 'rot_vel' and lambda_pose != 1.0:
            raise ValueError('lambda_pose is relevant only when training on velocities!')
        self.lambda_pose = lambda_pose
        self.lambda_orient = lambda_orient
        self.lambda_loc = lambda_loc
        self.lambda_rcxyz = lambda_rcxyz
        self.lambda_vel = lambda_vel
        self.lambda_root_vel = lambda_root_vel
        self.lambda_vel_rcxyz = lambda_vel_rcxyz
        self.lambda_fc = lambda_fc
        self.lambda_smpl_joints = lambda_smpl_joints
        self.lambda_smpl_vertices = lambda_smpl_vertices
        self.lambda_joints_consistency = lambda_joints_consistency
        self.lambda_joints_delta = lambda_joints_delta
        self.lambda_transl_delta = lambda_transl_delta
        self.lambda_orient_delta = lambda_orient_delta
        self.lambda_first_joints_delta = lambda_first_joints_delta
        self.lambda_first_transl_delta = lambda_first_transl_delta
        self.lambda_first_orient_delta = lambda_first_orient_delta
        self.lambda_rel_joints_delta = lambda_rel_joints_delta
        self.lambda_rel_transl_delta = lambda_rel_transl_delta
        self.lambda_rel_orient_delta = lambda_rel_orient_delta
        self.lambda_delta_norm = lambda_delta_norm
        self.lambda_smooth = lambda_smooth
        self.lambda_dct = lambda_dct
        self.lambda_jerk = lambda_jerk
        if self.lambda_rcxyz > 0.0 or self.lambda_vel > 0.0 or self.lambda_root_vel > 0.0 or (self.lambda_vel_rcxyz > 0.0) or (self.lambda_fc > 0.0):
            assert self.loss_type == LossType.MSE, 'Geometric losses are supported by MSE loss type only!'
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, 'betas must be 1-D'
        assert (betas > 0).all() and (betas <= 1).all()
        self.num_timesteps = int(betas.shape[0])
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        self.l2_loss = lambda a, b: (a - b) ** 2
        self.mse_loss = torch.nn.MSELoss(reduction='none')

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the dataset for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial dataset batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        assert posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] == x_start.shape[0]
        return (posterior_mean, posterior_variance, posterior_log_variance_clipped)

    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, previous_xstart=None, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x.shape[:2]
        assert t.shape == (B,)
        if not self.ignore_history and 'history_mask' in model_kwargs['y'].keys() and ('history_motion' in model_kwargs['y'].keys()):
            history_mask, history_motion = (model_kwargs['y']['history_mask'], model_kwargs['y']['history_motion'])
            assert self.model_mean_type == ModelMeanType.START_X, 'This feature supports only X_start pred for mow!'
            assert x.shape == history_mask.shape == history_motion.shape
            x = x * ~history_mask + history_motion * history_mask
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        if 'history_mask' in model_kwargs['y'].keys() and 'history_motion' in model_kwargs['y'].keys():
            history_mask, history_motion = (model_kwargs['y']['history_mask'], model_kwargs['y']['history_motion'])
            assert self.model_mean_type == ModelMeanType.START_X, 'This feature supports only X_start pred for mow!'
            assert x.shape == history_mask.shape == history_motion.shape
            model_output = model_output * ~history_mask + history_motion * history_mask
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {ModelVarType.FIXED_LARGE: (np.append(self.posterior_variance[1], self.betas[1:]), np.log(np.append(self.posterior_variance[1], self.betas[1:]))), ModelVarType.FIXED_SMALL: (self.posterior_variance, self.posterior_log_variance_clipped)}[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x
        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output))
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)
        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        return {'mean': model_mean, 'variance': model_variance, 'log_variance': model_log_variance, 'pred_xstart': pred_xstart}

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev - _extract_into_tensor(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape) * x_t

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = p_mean_var['mean'].float() + p_mean_var['variance'] * gradient.float()
        return new_mean

    def condition_mean_with_grad(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, t, p_mean_var, **model_kwargs)
        new_mean = p_mean_var['mean'].float() + p_mean_var['variance'] * gradient.float()
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        eps = self._predict_eps_from_xstart(x, t, p_mean_var['pred_xstart'])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        out = p_mean_var.copy()
        out['pred_xstart'] = self._predict_xstart_from_eps(x, t, eps)
        out['mean'], _, _ = self.q_posterior_mean_variance(x_start=out['pred_xstart'], x_t=x, t=t)
        return out

    def condition_score_with_grad(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        eps = self._predict_eps_from_xstart(x, t, p_mean_var['pred_xstart'])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, p_mean_var, **model_kwargs)
        out = p_mean_var.copy()
        out['pred_xstart'] = self._predict_xstart_from_eps(x, t, eps)
        out['mean'], _, _ = self.q_posterior_mean_variance(x_start=out['pred_xstart'], x_t=x, t=t)
        return out

    def p_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, const_noise=False):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs)
        noise = th.randn_like(x)
        if const_noise:
            noise = noise[[0]].repeat(x.shape[0], 1, 1, 1)
        nonzero_mask = (t != 0).float().view(-1, *[1] * (len(x.shape) - 1))
        if cond_fn is not None:
            out['mean'] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
        sample = out['mean'] + nonzero_mask * th.exp(0.5 * out['log_variance']) * noise
        return {'sample': sample, 'pred_xstart': out['pred_xstart']}

    def p_sample_with_grad(self, model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs)
            noise = th.randn_like(x)
            nonzero_mask = (t != 0).float().view(-1, *[1] * (len(x.shape) - 1))
            if cond_fn is not None:
                out['mean'] = self.condition_mean_with_grad(cond_fn, out, x, t, model_kwargs=model_kwargs)
        sample = out['mean'] + nonzero_mask * th.exp(0.5 * out['log_variance']) * noise
        return {'sample': sample, 'pred_xstart': out['pred_xstart'].detach()}

    def p_sample_loop(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False, skip_timesteps=0, init_image=None, randomize_class=False, cond_fn_with_grad=False, dump_steps=None, const_noise=False):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param const_noise: If True, will noise all samples with the same noise throughout sampling
        :return: a non-differentiable batch of samples.
        """
        final = None
        if dump_steps is not None:
            dump = []
        for i, sample in enumerate(self.p_sample_loop_progressive(model, shape, noise=noise, clip_denoised=clip_denoised, denoised_fn=denoised_fn, cond_fn=cond_fn, model_kwargs=model_kwargs, device=device, progress=progress, skip_timesteps=skip_timesteps, init_image=init_image, randomize_class=randomize_class, cond_fn_with_grad=cond_fn_with_grad, const_noise=const_noise)):
            if dump_steps is not None and i in dump_steps:
                dump.append(deepcopy(sample['sample']))
            final = sample
        if dump_steps is not None:
            return dump
        return final['sample']

    def p_sample_loop_progressive(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False, skip_timesteps=0, init_image=None, randomize_class=False, cond_fn_with_grad=False, const_noise=False):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)
        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]
        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes, size=model_kwargs['y'].shape, device=model_kwargs['y'].device)
            with th.no_grad():
                sample_fn = self.p_sample_with_grad if cond_fn_with_grad else self.p_sample
                out = sample_fn(model, img, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, cond_fn=cond_fn, model_kwargs=model_kwargs, const_noise=const_noise)
                yield out
                img = out['sample']

    def ddim_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, eta=0.0):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out_orig = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs)
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
        else:
            out = out_orig
        eps = self._predict_eps_from_xstart(x, t, out['pred_xstart'])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = eta * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        noise = th.randn_like(x)
        mean_pred = out['pred_xstart'] * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        nonzero_mask = (t != 0).float().view(-1, *[1] * (len(x.shape) - 1))
        sample = mean_pred + nonzero_mask * sigma * noise
        return {'sample': sample, 'pred_xstart': out_orig['pred_xstart']}

    def ddim_sample_with_grad(self, model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, eta=0.0):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out_orig = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs)
            if cond_fn is not None:
                out = self.condition_score_with_grad(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
            else:
                out = out_orig
        out['pred_xstart'] = out['pred_xstart'].detach()
        eps = self._predict_eps_from_xstart(x, t, out['pred_xstart'])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = eta * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        noise = th.randn_like(x)
        mean_pred = out['pred_xstart'] * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        nonzero_mask = (t != 0).float().view(-1, *[1] * (len(x.shape) - 1))
        sample = mean_pred + nonzero_mask * sigma * noise
        return {'sample': sample, 'pred_xstart': out_orig['pred_xstart'].detach()}

    def ddim_sample_loop(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False, eta=0.0, skip_timesteps=0, init_image=None, randomize_class=False, cond_fn_with_grad=False, dump_steps=None, const_noise=False):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        if dump_steps is not None:
            raise NotImplementedError()
        if const_noise == True:
            raise NotImplementedError()
        final = None
        for sample in self.ddim_sample_loop_progressive(model, shape, noise=noise, clip_denoised=clip_denoised, denoised_fn=denoised_fn, cond_fn=cond_fn, model_kwargs=model_kwargs, device=device, progress=progress, eta=eta, skip_timesteps=skip_timesteps, init_image=init_image, randomize_class=randomize_class, cond_fn_with_grad=cond_fn_with_grad):
            final = sample
        return final['sample']

    def ddim_sample_loop_progressive(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False, eta=0.0, skip_timesteps=0, init_image=None, randomize_class=False, cond_fn_with_grad=False):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)
        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]
        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes, size=model_kwargs['y'].shape, device=model_kwargs['y'].device)
            with th.no_grad():
                sample_fn = self.ddim_sample_with_grad if cond_fn_with_grad else self.ddim_sample
                out = sample_fn(model, img, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, cond_fn=cond_fn, model_kwargs=model_kwargs, eta=eta)
                yield out
                img = out['sample']

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
