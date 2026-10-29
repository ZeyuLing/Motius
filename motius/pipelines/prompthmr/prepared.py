"""Original image-to-track and temporal fusion graph on prepared video tracks."""
import torch
import numpy as np
from torch.amp import autocast
from torch.utils.data import DataLoader
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
from motius.datasets.prompthmr import PromptHMRVideoDataset
from motius.motion.representation.gvhmr.geometry import compute_cam_angvel
from motius.motion.representation.gvhmr.camera import get_bbx_xys_from_xyxy,normalize_kp2d
from motius.motion.representation.prompthmr.rotation import axis_angle_to_matrix
from .image import PromptHMRImagePrediction
from .video import PromptHMRVideoPrediction

class PromptHMRPreparedVideo:
    def __init__(self,bundle,geometry,body_model_path):
        self.model=PromptHMRImagePrediction(bundle,geometry)
        self.video=PromptHMRVideoPrediction(bundle,body_model_path)

    @torch.no_grad()
    def run(self, images, results, mask_prompt=True):
        tracks = results['people']
        camera = results['camera']
        img_focal = camera['img_focal']
        img_center = camera['img_center']
        cam_intrinsic = torch.eye(3)
        cam_intrinsic[0, 0] = img_focal.item()
        cam_intrinsic[1, 1] = img_focal.item()
        cam_intrinsic[0, 2] = img_center[0].item() if isinstance(img_center, np.ndarray) else img_center[0]
        cam_intrinsic[1, 2] = img_center[1].item() if isinstance(img_center, np.ndarray) else img_center[1]
        dataset = PromptHMRVideoDataset(images, tracks, cam_intrinsic)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=lambda x: x)
        for k, v in tracks.items():
            tracks[k]['smplx_pose'] = []
            tracks[k]['smplx_transl'] = []
            tracks[k]['smplx_betas'] = []
            tracks[k]['prhmr_img_feats'] = []
        for batch in dataloader:
            with autocast('cuda'):
                output = self.model(batch, mask_prompt=mask_prompt)
            for bid in range(len(batch)):
                track_id = batch[bid]['track_ids']
                for tid in track_id:
                    btid = track_id.index(tid)
                    tracks[tid]['smplx_pose'].append(output[bid]['rotmat'][btid])
                    tracks[tid]['smplx_transl'].append(output[bid]['transl'][btid])
                    tracks[tid]['smplx_betas'].append(output[bid]['betas'][btid])
                    tracks[tid]['prhmr_img_feats'].append(output[bid]['features'][btid])
        for k, v in tracks.items():
            tracks[k]['smplx_pose'] = torch.stack(tracks[k]['smplx_pose']).float()
            tracks[k]['smplx_transl'] = torch.stack(tracks[k]['smplx_transl']).float()
            tracks[k]['smplx_betas'] = torch.stack(tracks[k]['smplx_betas']).float()
            tracks[k]['prhmr_img_feats'] = torch.stack(tracks[k]['prhmr_img_feats']).float()
        print(f'Running PRHMR-Vid for tracks')
        for idx, k in enumerate(list(tracks.keys())):
            seqlen = tracks[k]['prhmr_img_feats'].shape[0]
            R_w2c = torch.eye(3).repeat(seqlen, 1, 1)
            cam_angvel = compute_cam_angvel(R_w2c)
            bbx_xys = get_bbx_xys_from_xyxy(torch.from_numpy(tracks[k]['bboxes'])).numpy()
            smoothed = np.array([signal.medfilt(param, 11) for param in bbx_xys.T]).T
            bbx_xys = np.array([gaussian_filter1d(traj, 3) for traj in smoothed.T]).T
            bbx_xys = torch.from_numpy(bbx_xys).float()
            vitpose = torch.from_numpy(tracks[k]['vitpose'])
            vitpose = normalize_kp2d(vitpose, bbx_xys).float()
            batch = {'length': torch.tensor([seqlen]), 'obs': vitpose[None], 'bbx_xys': bbx_xys[None], 'K_fullimg': cam_intrinsic[None, None].repeat_interleave(seqlen, 1), 'cam_angvel': cam_angvel[None], 'f_imgseq': tracks[k]['prhmr_img_feats'][None]}
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.amp.autocast('cuda'):
                prhmr_vid_output_w_kpts = self.video.forward(batch, static_cam=True)
            batch['obs'] = torch.zeros_like(batch['obs'])
            with torch.amp.autocast('cuda'):
                prhmr_vid_output = self.video.forward(batch, static_cam=True)
            contact_label = prhmr_vid_output['model_output']['static_conf_logits'].sigmoid() > 0.8
            smplx_pose_aa = torch.cat([prhmr_vid_output['pred_smpl_params_incam']['global_orient'][0], prhmr_vid_output['pred_smpl_params_incam']['body_pose'][0], torch.zeros(seqlen, 9).cuda()], dim=-1)
            rotmat = axis_angle_to_matrix(smplx_pose_aa.reshape(-1, 3)).reshape(-1, 25, 3, 3)
            hand_pose_rotmat = torch.zeros(rotmat.shape[0], 30, 3, 3).to(rotmat)
            rotmat = torch.cat([rotmat, hand_pose_rotmat], dim=1)
            hps_results = {'rotmat': rotmat.cpu().float().numpy(), 'pose': smplx_pose_aa.cpu().numpy(), 'shape': prhmr_vid_output['pred_smpl_params_incam']['betas'][0].cpu().float().numpy(), 'trans': prhmr_vid_output_w_kpts['pred_smpl_params_incam']['transl'][0].cpu().float().numpy(), 'contact': contact_label[0].cpu().numpy(), 'static_conf_logits': prhmr_vid_output['model_output']['static_conf_logits'][0].cpu().numpy()}
            results['people'][k]['smplx_cam'] = hps_results
        return results
