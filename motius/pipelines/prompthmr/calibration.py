"""Original SPEC distribution-to-angle decoding outside neural modules."""
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm

def _softmax(tensor, temperature, dim=-1):
    return F.softmax(tensor * temperature, dim=dim)

def softargmax1d(heatmaps, temperature=None, normalize_keypoints=True):
    dtype, device = (heatmaps.dtype, heatmaps.device)
    if temperature is None:
        temperature = torch.tensor(1.0, dtype=dtype, device=device)
    batch_size, num_channels, dim = heatmaps.shape
    points = torch.arange(0, dim, device=device, dtype=dtype).reshape(1, 1, dim).expand(batch_size, -1, -1)
    normalized_heatmap = _softmax(heatmaps.reshape(batch_size, num_channels, -1), temperature=temperature.reshape(1, -1, 1), dim=-1)
    keypoints = (normalized_heatmap.reshape(batch_size, -1, dim) * points).sum(dim=-1)
    if normalize_keypoints:
        keypoints = keypoints / (dim - 1) * 2 - 1
    return (keypoints, normalized_heatmap.reshape(batch_size, -1, dim))

def get_bins(minval, maxval, sigma, alpha, beta, kappa):
    """Remember, bin 0 = below value! last bin mean >= maxval"""
    x = np.linspace(minval, maxval, 255)
    rv = norm(0, sigma)
    pdf = rv.pdf(x)
    pdf /= pdf.max()
    pdf *= alpha
    pdf = pdf.max() * beta - pdf
    cumsum = np.cumsum(pdf)
    cumsum = cumsum / cumsum.max() * kappa
    cumsum -= cumsum[pdf.size // 2]
    return cumsum

pitch_bins = np.linspace(-0.6, 0.6, 255)
pitch_bins_centers = pitch_bins.copy()
pitch_bins_centers[:-1] += np.diff(pitch_bins_centers)/2
pitch_bins_centers = np.append(pitch_bins_centers, pitch_bins[-1])

horizon_bins = np.linspace(-0.5, 1.5, 255)
horizon_bins_centers = horizon_bins.copy()
horizon_bins_centers[:-1] += np.diff(horizon_bins_centers)/2
horizon_bins_centers = np.append(horizon_bins_centers, horizon_bins[-1])

roll_bins = get_bins(-np.pi/6, np.pi/6, 0.5, 0.04, 1.1, np.pi)
# roll_bins = get_bins(-np.pi/6, np.pi/6, 0.2, 0.04, 1.1, np.pi/3)
roll_bins_centers = roll_bins.copy()
roll_bins_centers[:-1] += np.diff(roll_bins_centers)/2
roll_bins_centers = np.append(roll_bins_centers, roll_bins[-1])

vfov_bins = np.linspace(0.2617, 2.1, 255)
vfov_bins_centers = vfov_bins.copy()
vfov_bins_centers[:-1] += np.diff(vfov_bins_centers)/2
vfov_bins_centers = np.append(vfov_bins_centers, vfov_bins[-1])

roll_new_bins = np.linspace(-0.6, 0.6, 255)
roll_new_bins_centers = roll_new_bins.copy()
roll_new_bins_centers[:-1] += np.diff(roll_new_bins_centers)/2
roll_new_bins_centers = np.append(roll_new_bins_centers, roll_new_bins[-1])


def bins2pitch(bins):
    if isinstance(bins, torch.Tensor):
        bins = bins.cpu().numpy()
    idxes = np.argmax(bins, axis=bins.ndim - 1)
    return pitch_bins_centers[idxes]

def bins2roll(bins):
    if isinstance(bins, torch.Tensor):
        bins = bins.cpu().numpy()
    idxes = np.argmax(bins, axis=bins.ndim - 1)
    return roll_bins_centers[idxes]

def bins2vfov(bins):
    if isinstance(bins, torch.Tensor):
        bins = bins.cpu().numpy()
    idxes = np.argmax(bins, axis=bins.ndim - 1)
    return vfov_bins_centers[idxes]

def soft_idx_to_angle(soft_idx, min, max):
    return (max - min) * ((soft_idx + 1) / 2) + min

def get_softargmax(pred):
    pred = pred.unsqueeze(1)
    pred_argmax, _ = softargmax1d(pred, normalize_keypoints=True)
    pred_argmax = pred_argmax.reshape(-1)
    return pred_argmax

@torch.no_grad()
def convert_preds_to_angles(pred_vfov, pred_pitch, pred_roll, loss_type='kl', return_type='torch', legacy=False):
    if loss_type in ('kl', 'ce'):
        pred_vfov = bins2vfov(pred_vfov)
        pred_pitch = bins2pitch(pred_pitch)
        pred_roll = bins2roll(pred_roll)
    elif loss_type in ('softargmax_l2', 'softargmax_biased_l2'):
        pred_vfov = soft_idx_to_angle(get_softargmax(pred_vfov), min=np.min(vfov_bins), max=np.max(vfov_bins))
        pred_pitch = soft_idx_to_angle(get_softargmax(pred_pitch), min=np.min(pitch_bins), max=np.max(pitch_bins))
        if not legacy:
            pred_roll = soft_idx_to_angle(get_softargmax(pred_roll), min=-0.6, max=0.6)
        else:
            pred_roll = bins2roll(pred_roll)
    if return_type == 'np' and isinstance(pred_vfov, torch.Tensor):
        return (pred_vfov.cpu().numpy(), pred_pitch.cpu().numpy(), pred_roll.cpu().numpy())
    if return_type == 'torch' and isinstance(pred_vfov, np.ndarray):
        return (torch.from_numpy(pred_vfov), torch.from_numpy(pred_pitch), torch.from_numpy(pred_roll))
    return (pred_vfov, pred_pitch, pred_roll)


@torch.no_grad()
def calibrate_video(bundle,images,stride=1):
    from motius.datasets.prompthmr import CalibrationImageDataset
    selected=images[::stride]
    dataset=CalibrationImageDataset(selected,min_size=min(selected[0].shape[:2]))
    device=next(bundle.parameters()).device
    output={};focals=[]
    for index,batch in enumerate(dataset):
        distributions=bundle(batch['img'].unsqueeze(0).to(device).float())
        angles=convert_preds_to_angles(*distributions,loss_type='softargmax_l2')
        vfov,pitch,roll=[value.detach().cpu().numpy().squeeze() for value in angles]
        original_width,original_height=batch['orig_shape']
        focal=original_height/2./np.tan(vfov/2.)
        result=dict(vfov=vfov.item(),f_pix=focal,pitch=pitch.item(),roll=roll.item())
        output[batch['imgname']]=result
        if index==0:output['first_frame']=dict(result)
        focals.append(focal)
    for name,function in [('avg',np.mean),('min',np.min),('max',np.max),('std',np.std),('median',np.median)]:
        output[name+'_focal_length']=function(focals)
    return output
