import torch
from motius.motion.representation import dart_rotation as transforms
from motius.motion.representation.dart import get_smplx_param_from_6d

def denoiser_loss(self, motion, cond, history_motion, future_motion_gt, future_motion_pred, latent_gt, latent_pred, weights):
    train_args = self.args.train_args
    model_kwargs = cond
    future_length = self.train_dataset.future_length
    history_length = self.train_dataset.history_length
    num_primitive = self.train_dataset.num_primitive
    terms = {}
    feature_rec_loss = self.rec_criterion(future_motion_pred, future_motion_gt)
    terms['feature_rec'] = feature_rec_loss
    latent_rec_loss = self.rec_criterion(latent_pred, latent_gt)
    terms['latent_rec'] = latent_rec_loss
    'smplx consistency losses'
    dataset = self.train_dataset
    primitive_utility = dataset.primitive_utility
    if train_args.weight_joints_consistency > 0 or train_args.weight_smpl_joints_rec > 0:
        gt_motion_tensor = future_motion_gt
        pred_motion_tensor = future_motion_pred
        genders = model_kwargs['y']['gender']
        betas = model_kwargs['y']['betas']

        def get_smpl_body(motion_tensor, genders, betas):
            batch_size, num_frames, _ = motion_tensor.shape
            device = motion_tensor.device
            smpl_joints = []
            smpl_vertices = []
            joints = []
            for gender_name in ['female', 'male']:
                body_model = primitive_utility.get_smpl_model(gender=gender_name)
                gender_idx = [idx for idx in range(len(genders)) if genders[idx] == gender_name]
                sub_batch_size = len(gender_idx)
                if len(gender_idx) == 0:
                    continue
                gender_betas = betas[gender_idx, history_length:, :].reshape(sub_batch_size * num_frames, 10)
                gender_motion_tensor = motion_tensor[gender_idx, :, :]
                gender_motion_tensor = dataset.denormalize(gender_motion_tensor).reshape(sub_batch_size * num_frames, -1)
                motion_dict = dataset.tensor_to_dict(gender_motion_tensor)
                motion_dict.update({'betas': gender_betas})
                joints.append(motion_dict['joints'].reshape(sub_batch_size, num_frames, 22, 3))
                smplx_param = get_smplx_param_from_6d(motion_dict)
                smplxout = body_model(return_verts=False, **smplx_param)
                smpl_joints.append(smplxout.joints[:, :22, :].reshape(sub_batch_size, num_frames, 22, 3))
            smpl_joints = torch.cat(smpl_joints, dim=0)
            smpl_vertices = None
            joints = torch.cat(joints, dim=0)
            return {'smpl_joints': smpl_joints, 'smpl_vertices': smpl_vertices, 'joints': joints}
        with torch.no_grad():
            gt_result_dict = get_smpl_body(gt_motion_tensor, genders, betas)
        pred_result_dict = get_smpl_body(pred_motion_tensor, genders, betas)
        terms['smpl_joints_rec'] = self.rec_criterion(pred_result_dict['smpl_joints'], gt_result_dict['smpl_joints'])
        terms['joints_consistency'] = self.rec_criterion(pred_result_dict['joints'], pred_result_dict['smpl_joints'])
    else:
        terms['smpl_joints_rec'] = torch.zeros(1, device=self.device)
        terms['joints_consistency'] = torch.zeros(1, device=self.device)
    'temporal delta loss'
    pred_motion_tensor = torch.cat([history_motion[:, -1:, :], future_motion_pred], dim=1)
    pred_motion_tensor = dataset.denormalize(pred_motion_tensor)
    pred_feature_dict = dataset.tensor_to_dict(pred_motion_tensor)
    pred_joints_delta = pred_feature_dict['joints_delta'][:, :-1, :]
    pred_transl_delta = pred_feature_dict['transl_delta'][:, :-1, :]
    pred_orient_delta = pred_feature_dict['global_orient_delta_6d'][:, :-1, :]
    calc_joints_delta = pred_feature_dict['joints'][:, 1:, :] - pred_feature_dict['joints'][:, :-1, :]
    calc_transl_delta = pred_feature_dict['transl'][:, 1:, :] - pred_feature_dict['transl'][:, :-1, :]
    pred_orient = transforms.rotation_6d_to_matrix(pred_feature_dict['poses_6d'][:, :, :6])
    calc_orient_delta_matrix = torch.matmul(pred_orient[:, 1:], pred_orient[:, :-1].permute(0, 1, 3, 2))
    calc_orient_delta_6d = transforms.matrix_to_rotation_6d(calc_orient_delta_matrix)
    terms['joints_delta'] = self.rec_criterion(calc_joints_delta, pred_joints_delta)
    terms['transl_delta'] = self.rec_criterion(calc_transl_delta, pred_transl_delta)
    terms['orient_delta'] = self.rec_criterion(calc_orient_delta_6d, pred_orient_delta)
    loss = train_args.weight_latent_rec * latent_rec_loss + train_args.weight_feature_rec * feature_rec_loss + train_args.weight_smpl_joints_rec * terms['smpl_joints_rec'] + train_args.weight_joints_consistency * terms['joints_consistency'] + train_args.weight_joints_delta * terms['joints_delta'] + train_args.weight_transl_delta * terms['transl_delta'] + train_args.weight_orient_delta * terms['orient_delta']
    terms['loss'] = loss
    return terms

def vae_loss(self, motion, cond, history_motion, future_motion_gt, future_motion_pred, latent, dist):
    dataset = self.train_dataset
    train_args = self.args.train_args
    model_kwargs = cond
    future_length = self.train_dataset.future_length
    history_length = self.train_dataset.history_length
    num_primitive = self.train_dataset.num_primitive
    terms = {}
    mu_ref = torch.zeros_like(dist.loc)
    scale_ref = torch.ones_like(dist.scale)
    dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
    kl_loss = torch.distributions.kl_divergence(dist, dist_ref)
    kl_loss = kl_loss.mean()
    terms['kl_loss'] = kl_loss
    rec_loss = self.rec_criterion(future_motion_pred, future_motion_gt)
    terms['rec_loss'] = rec_loss
    'smplx consistency losses'
    if train_args.weight_smpl_joints_rec > 0 or train_args.weight_joints_consistency > 0:
        gt_motion_tensor = future_motion_gt
        pred_motion_tensor = future_motion_pred
        genders = model_kwargs['y']['gender']
        betas = model_kwargs['y']['betas']
        dataset = self.train_dataset
        primitive_utility = dataset.primitive_utility

        def get_smpl_body(motion_tensor, genders, betas):
            batch_size, num_frames, _ = motion_tensor.shape
            device = motion_tensor.device
            smpl_joints = []
            smpl_vertices = []
            joints = []
            for gender_name in ['female', 'male']:
                body_model = primitive_utility.get_smpl_model(gender=gender_name)
                gender_idx = [idx for idx in range(len(genders)) if genders[idx] == gender_name]
                sub_batch_size = len(gender_idx)
                if len(gender_idx) == 0:
                    continue
                gender_betas = betas[gender_idx, history_length:, :].reshape(sub_batch_size * num_frames, 10)
                gender_motion_tensor = motion_tensor[gender_idx, :, :]
                gender_motion_tensor = dataset.denormalize(gender_motion_tensor).reshape(sub_batch_size * num_frames, -1)
                motion_dict = dataset.tensor_to_dict(gender_motion_tensor)
                motion_dict.update({'betas': gender_betas})
                joints.append(motion_dict['joints'].reshape(sub_batch_size, num_frames, 22, 3))
                smplx_param = get_smplx_param_from_6d(motion_dict)
                smplxout = body_model(return_verts=False, **smplx_param)
                smpl_joints.append(smplxout.joints[:, :22, :].reshape(sub_batch_size, num_frames, 22, 3))
            smpl_joints = torch.cat(smpl_joints, dim=0)
            smpl_vertices = None
            joints = torch.cat(joints, dim=0)
            return {'smpl_joints': smpl_joints, 'smpl_vertices': smpl_vertices, 'joints': joints}
        with torch.no_grad():
            gt_result_dict = get_smpl_body(gt_motion_tensor, genders, betas)
        pred_result_dict = get_smpl_body(pred_motion_tensor, genders, betas)
        terms['smpl_joints_rec'] = self.rec_criterion(pred_result_dict['smpl_joints'], gt_result_dict['smpl_joints'])
        terms['joints_consistency'] = self.rec_criterion(pred_result_dict['joints'], pred_result_dict['smpl_joints'])
    else:
        terms['smpl_joints_rec'] = torch.zeros(1, device=self.device)
        terms['joints_consistency'] = torch.zeros(1, device=self.device)
    'temporal delta loss'
    pred_motion_tensor = torch.cat([history_motion[:, -1:, :], future_motion_pred], dim=1)
    pred_motion_tensor = dataset.denormalize(pred_motion_tensor)
    pred_feature_dict = dataset.tensor_to_dict(pred_motion_tensor)
    pred_joints_delta = pred_feature_dict['joints_delta'][:, :-1, :]
    pred_transl_delta = pred_feature_dict['transl_delta'][:, :-1, :]
    pred_orient_delta = pred_feature_dict['global_orient_delta_6d'][:, :-1, :]
    calc_joints_delta = pred_feature_dict['joints'][:, 1:, :] - pred_feature_dict['joints'][:, :-1, :]
    calc_transl_delta = pred_feature_dict['transl'][:, 1:, :] - pred_feature_dict['transl'][:, :-1, :]
    pred_orient = transforms.rotation_6d_to_matrix(pred_feature_dict['poses_6d'][:, :, :6])
    calc_orient_delta_matrix = torch.matmul(pred_orient[:, 1:], pred_orient[:, :-1].permute(0, 1, 3, 2))
    calc_orient_delta_6d = transforms.matrix_to_rotation_6d(calc_orient_delta_matrix)
    terms['joints_delta'] = self.rec_criterion(calc_joints_delta, pred_joints_delta)
    terms['transl_delta'] = self.rec_criterion(calc_transl_delta, pred_transl_delta)
    terms['orient_delta'] = self.rec_criterion(calc_orient_delta_6d, pred_orient_delta)
    loss = train_args.weight_kl * kl_loss + train_args.weight_rec * rec_loss + train_args.weight_smpl_joints_rec * terms['smpl_joints_rec'] + train_args.weight_joints_consistency * terms['joints_consistency'] + train_args.weight_joints_delta * terms['joints_delta'] + train_args.weight_transl_delta * terms['transl_delta'] + train_args.weight_orient_delta * terms['orient_delta']
    terms['loss'] = loss
    return terms
