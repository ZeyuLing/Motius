"""Native map, keyframe and bundle-adjustment composition around DROID networks."""
import torch
from .depth_video import DepthVideo
from .motion_filter import MotionFilter
from .droid_frontend import DroidFrontend
from .droid_backend import DroidBackend
from .trajectory_filler import PoseTrajectoryFiller

class Droid:
    def __init__(self,bundle,args):
        self.net=bundle
        self.args=args
        self.disable_vis=True
        self.video=DepthVideo(args.image_size,args.buffer,stereo=args.stereo,opt_intr=args.opt_intr,camera_model=args.camera_model)
        self.filterx=MotionFilter(bundle,self.video,thresh=args.filter_thresh)
        self.frontend=DroidFrontend(bundle,self.video,args)
        self.backend=DroidBackend(bundle,self.video,args)
        self.traj_filler=PoseTrajectoryFiller(bundle,self.video)

    def track(self, tstamp, image, depth=None, intrinsics=None, mask=None):
        """ main thread - update map """
        with torch.no_grad():
            self.filterx.track(tstamp, image, depth, intrinsics, mask)
            self.frontend()

    def has_motion(self, tstamp, image, thresh=None):
        with torch.no_grad():
            return self.filterx.has_motion(tstamp, image, thresh)

    def terminate(self, stream=None, backend=True):
        """ terminate the visualization process, return poses [t, q] """
        del self.frontend
        if backend:
            torch.cuda.empty_cache()
            self.backend(7)
            torch.cuda.empty_cache()
            self.backend(12)
        camera_trajectory = self.traj_filler(stream)
        intr = self.video.intrinsics[0, :].clone()
        intr[:4] *= 8.0
        return (camera_trajectory.inv().data.cpu().numpy(), intr.data.cpu().numpy())

    def compute_error(self):
        """ compute slam reprojection error """
        del self.frontend
        torch.cuda.empty_cache()
        self.backend(12)
        return self.backend.errors[-1]
