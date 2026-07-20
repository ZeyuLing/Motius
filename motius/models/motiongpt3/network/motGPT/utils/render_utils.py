
import os
import torch
import time
import imageio
import numpy as np
import moviepy.editor as mp
from scipy.spatial.transform import Rotation as RRR
import motius.models.motiongpt3.network.motGPT.render.matplot.plot_3d_global as plot_3d
from motius.models.motiongpt3.network.motGPT.render.pyrender.hybrik_loc2rot import HybrIKJointsToRotmat
from motius.models.motiongpt3.network.motGPT.render.pyrender.smpl_render import SMPLRender

SMPL_MODEL_PATH = 'deps/smpl_models/smpl'

def render_motion(data, feats, output_dir, fname=None, method='fast', smpl_model_path=SMPL_MODEL_PATH, fps=20):
    if fname is None:
        fname = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(
            time.time())) + str(np.random.randint(10000, 99999))
    video_fname = fname + '.mp4'
    feats_fname = fname + '.npy'
    output_npy_path = os.path.join(output_dir, feats_fname)
    output_mp4_path = os.path.join(output_dir, video_fname)
    # np.save(output_npy_path, feats)

    if method == 'slow':
        if len(data.shape) == 4:
            data = data[0]
        data = data - data[0, 0]
        pose_generator = HybrIKJointsToRotmat()
        pose = pose_generator(data)
        pose = np.concatenate([
            pose,
            np.stack([np.stack([np.eye(3)] * pose.shape[0], 0)] * 2, 1)
        ], 1)
        shape = [768, 768]
        render = SMPLRender(smpl_model_path)

        r = RRR.from_rotvec(np.array([np.pi, 0.0, 0.0]))
        pose[:, 0] = np.matmul(r.as_matrix().reshape(1, 3, 3), pose[:, 0])
        vid = []
        aroot = data[:, 0].copy()
        aroot[:, 1] = -aroot[:, 1]
        aroot[:, 2] = -aroot[:, 2]
        params = dict(pred_shape=np.zeros([1, 10]),
                      pred_root=aroot,
                      pred_pose=pose)
        render.init_renderer([shape[0], shape[1], 3], params)
        for i in range(data.shape[0]):
            renderImg = render.render(i)
            vid.append(renderImg)

        # out = np.stack(vid, axis=0)
        out_video = mp.ImageSequenceClip(vid, fps=fps)
        out_video.write_videofile(output_mp4_path, fps=fps)
        del render

    elif method == 'fast':
        output_gif_path = output_mp4_path[:-4] + '.gif'
        if len(data.shape) == 3:
            data = data[None]
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        pose_vis = plot_3d.draw_to_batch(data, [''], None, fps=fps)[0].cpu().numpy()

        out_video = mp.ImageSequenceClip(list(pose_vis),fps=fps)
        out_video.write_videofile(output_mp4_path, fps=fps)
        # out_video = mp.VideoClip(make_frame=lambda t:pose_vis[int(t*fps)], duration=len(pose_vis)/fps)
        # out_video.write_videofile(output_mp4_path,fps=fps)
        del pose_vis
