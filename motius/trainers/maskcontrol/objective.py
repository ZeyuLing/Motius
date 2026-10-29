"""Published MaskControl objective and conditioning draws."""
from motius.motion.conditioning.clip_text import encode_clip_text
import torch
import torch.nn.functional as F
from einops import repeat
from motius.motion.conditioning.maskcontrol import CONTROL_JOINT_IDS as control_joint_ids
from motius.motion.conditioning.momask_tokens import lengths_to_mask, uniform, cosine_schedule, get_mask_subset_prob
from motius.motion.representation.humanml import recover_from_ric
from motius.trainers.momask.objective import cal_performance

def compute_control_objective(network, predict, vq_model, mean, std, mask_emb_vq, control, y, m_length, pose):
    x_encoder = vq_model.encoder(pose.permute(0, 2, 1))
    code_idx = vq_model.quantizer.quantize(x_encoder)
    ids = code_idx[..., 0]
    assert m_length.min() > 30
    m_lens = m_length // 4
    global_joint = recover_from_ric((pose * std + mean).float(), network.opt.joints_num)
    bs, ntokens = ids.shape
    device = ids.device
    if control == 'trajectory' or control == 'random':
        rand_length = torch.rand(m_length.shape).to(pose.device) * 196
        rand_length = rand_length.round().clamp(min=1, max=196)
        all_len_mask = lengths_to_mask(m_length, pose.shape[1])
        batch_randperm = torch.rand((bs, pose.shape[1]), device=pose.device)
        batch_randperm[~all_len_mask] = 1
        batch_randperm = batch_randperm.argsort(dim=-1)
        global_joint_mask = batch_randperm < rand_length.unsqueeze(-1)
        global_joint_mask = global_joint_mask * all_len_mask
        if control == 'trajectory':
            global_joint_mask = repeat(global_joint_mask, 'b f -> b f j', j=network.opt.joints_num).clone()
            global_joint_mask[..., 1:] = False
        elif control == 'random':
            _global_joint_mask = global_joint_mask
            global_joint_mask = torch.zeros((*global_joint_mask.shape, network.opt.joints_num), device=_global_joint_mask.device, dtype=bool)
            control_joints = torch.tensor(control_joint_ids, device=pose.device)
            rand_indx = torch.randint(len(control_joints), (_global_joint_mask.shape[0],))
            global_joint_mask[torch.arange(global_joint_mask.shape[0]), :, control_joints[rand_indx]] = _global_joint_mask
    elif control == 'cross':
        rand_length = torch.rand(m_length.shape, device=m_length.device) * m_length * len(control_joint_ids)
        rand_length = rand_length.to(pose.device).round().clamp(min=1)
        all_len_mask = lengths_to_mask(m_length, pose.shape[1])
        batch_randperm = torch.rand((*all_len_mask.shape, len(control_joint_ids)), device=pose.device)
        batch_randperm[~all_len_mask] = 1
        batch_randperm = batch_randperm.reshape((all_len_mask.shape[0], -1))
        batch_randperm = batch_randperm.argsort(dim=-1)
        batch_randperm = batch_randperm.reshape((*all_len_mask.shape, -1))
        _global_joint_mask = batch_randperm < rand_length.unsqueeze(-1).unsqueeze(-1)
        _global_joint_mask = _global_joint_mask * all_len_mask.unsqueeze(-1)
        global_joint_mask = torch.zeros((*pose.shape[:2], network.opt.joints_num), device=pose.device, dtype=bool)
        global_joint_mask[..., control_joint_ids] = _global_joint_mask
    else:
        raise Exception(f'{control} is not implemented yet!!!')
    non_pad_mask = lengths_to_mask(m_lens, ntokens)
    ids = torch.where(non_pad_mask, ids, network.pad_id)
    with torch.no_grad():
        cond_vector = encode_clip_text(network, y)
    rand_time = uniform((bs,), device=device)
    rand_mask_probs = cosine_schedule(rand_time)
    num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)
    batch_randperm = torch.rand((bs, ntokens), device=device).argsort(dim=-1)
    mask = batch_randperm < num_token_masked.unsqueeze(-1)
    mask &= non_pad_mask
    labels = torch.where(non_pad_mask, ids, network.mask_id)
    x_ids = ids.clone()
    mask_rid = get_mask_subset_prob(mask, 0.1)
    rand_id = torch.randint_like(x_ids, high=network.opt.num_tokens)
    x_ids = torch.where(mask_rid, rand_id, x_ids)
    mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
    x_ids = torch.where(mask_mid, network.mask_id, x_ids)
    _x_ids = x_ids.clone()
    id_overflow = _x_ids >= network.opt.num_tokens
    _x_ids[id_overflow] = 0
    x_d = vq_model.quantizer.get_codes_from_indices(_x_ids.unsqueeze(-1))
    x = x_d.sum(dim=0)
    x[id_overflow] = mask_emb_vq
    padding_mask = ~non_pad_mask
    x = x.permute(0, 2, 1)
    pred_motions = vq_model.decoder(x)
    pred_motions_denorm = pred_motions * std + mean
    pred_motions_denorm = recover_from_ric(pred_motions_denorm.float(), network.opt.joints_num)
    ctrlNet_cond = (global_joint - pred_motions_denorm) * global_joint_mask.unsqueeze(-1)
    ctrlNet_cond2 = global_joint * global_joint_mask.unsqueeze(-1)
    if control == 'trajectory':
        ctrlNet_cond = ctrlNet_cond[:, :, 0]
        ctrlNet_cond2 = ctrlNet_cond2[:, :, 0]
    else:
        ctrlNet_cond = ctrlNet_cond[..., control_joint_ids, :]
        ctrlNet_cond = ctrlNet_cond.reshape((*ctrlNet_cond.shape[:2], -1))
        ctrlNet_cond2 = ctrlNet_cond2[..., control_joint_ids, :]
        ctrlNet_cond2 = ctrlNet_cond2.reshape((*ctrlNet_cond2.shape[:2], -1))
    ctrlNet_cond = torch.cat([ctrlNet_cond, ctrlNet_cond2], dim=-1)
    logits = predict(x_ids, cond_vector, ~non_pad_mask, force_mask=False, control_condition=ctrlNet_cond)
    ce_loss, pred_id, acc = cal_performance(logits, labels, ignore_index=network.mask_id)
    emb = F.softmax(logits.permute(0, 2, 1) / 1, dim=-1) @ vq_model.quantizer.codebooks[0]
    emb = emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)
    pred_motions = vq_model.forward_decoder(emb)
    pred_motions_denorm = pred_motions * std + mean
    pred_motions_denorm = recover_from_ric(pred_motions_denorm.float(), network.opt.joints_num)
    loss_emb = torch.zeros(1, device=emb.device)
    loss_tta = F.l1_loss(pred_motions_denorm[global_joint_mask], global_joint[global_joint_mask], reduction='mean')
    return (loss_emb, ce_loss, pred_id, acc, loss_tta)
