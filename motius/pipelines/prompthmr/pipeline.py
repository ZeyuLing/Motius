"""Native PromptHMR camera reconstruction through Motius neural bundles."""
from pathlib import Path
import json,shutil,tempfile
from collections.abc import Mapping
import numpy as np
from motius.configuration.prompthmr import sha256_file
from motius.models.prompthmr import PromptHMRBundle
from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES
from .parser import parse_prompthmr_results


@PIPELINES.register_module()
class PromptHMRPipeline(BasePipeline):
    BUNDLE_CLS='motius.models.prompthmr.PromptHMRBundle'

    def __init__(self,bundle,*,body_models_root=None,options=None):
        super().__init__(bundle)
        self.body_models_root=body_models_root
        saved=bundle.artifact_root/'prompthmr_pipeline.json' if bundle.artifact_root is not None else None
        self.options=json.loads(saved.read_text()).get('options',{}) if saved is not None and saved.is_file() else {}
        self.options.update(options or {})
        self._capture=None
        self._hashes=None

    def checkpoint_hashes(self):
        if self._hashes is None:
            root=self.bundle.artifact_root
            if root is None:raise ValueError('Checkpoint provenance requires a pretrained artifact')
            if (root/'prompthmr_networks.pt').is_file():
                self._hashes=dict(video_head=sha256_file(root/'prompthmr_networks.pt'))
            else:
                video=root/'video/phmr_b1b2.ckpt'
                if not video.is_file():video=root/'video/prhmr_release_002.ckpt'
                self._hashes=dict(video_head=sha256_file(video),image_model=sha256_file(root/'checkpoint.ckpt'))
        return dict(self._hashes)

    def parse_output(self,source,*,original_fps,output_fps=None,checkpoint_sha256=None,checkpoint_sha256s=None):
        hashes=dict(checkpoint_sha256s or {})
        if checkpoint_sha256 is None:
            hashes.update(self.checkpoint_hashes());checkpoint_sha256=hashes['video_head']
        return parse_prompthmr_results(source,checkpoint_sha256=checkpoint_sha256,original_fps=original_fps,
            output_fps=output_fps,checkpoint_sha256s=hashes)

    def replay_licensed_geometry(self,result,model_path,**kwargs):
        from .replay import replay_prompthmr_with_licensed_model
        return replay_prompthmr_with_licensed_model(result,Path(model_path),**kwargs)

    def capture(self):
        if self._capture is None:
            if self.body_models_root is None:raise ValueError('Provide body_models_root containing licensed smpl/ and smplx/ assets')
            from .capture import PromptHMRCapture
            self._capture=PromptHMRCapture(self.bundle,self.body_models_root,options=self.options)
        return self._capture

    def infer_monocular_motion_capture(self,input_video,*,output_dir=None,original_fps=None,output_fps=None,
                                      static_camera=False,max_frames=None,return_raw=False,deterministic=False,seed=None):
        video=Path(input_video).expanduser().resolve()
        if not video.is_file():raise FileNotFoundError(video)
        if output_dir is None:
            with tempfile.TemporaryDirectory(prefix='motius-prompthmr-') as temporary:
                return self.infer_monocular_motion_capture(video,output_dir=temporary,original_fps=original_fps,
                    output_fps=output_fps,static_camera=static_camera,max_frames=max_frames,return_raw=return_raw,
                    deterministic=deterministic,seed=seed)
        import torch,os,random
        if seed is not None:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        settings=(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic,
            torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        try:
            if deterministic:
                os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
                torch.use_deterministic_algorithms(True)
                torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
                torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            result,_,fps=self.capture().infer_video(video,output_dir,static_camera=static_camera,max_frames=max_frames)
        finally:
            torch.use_deterministic_algorithms(settings[0],warn_only=settings[1])
            torch.backends.cudnn.benchmark=settings[2];torch.backends.cudnn.deterministic=settings[3]
            torch.backends.cuda.matmul.allow_tf32=settings[4];torch.backends.cudnn.allow_tf32=settings[5]
        import joblib
        joblib.dump(result,Path(output_dir)/'results.pkl')
        if return_raw:return result
        if original_fps is None:
            import cv2
            reader=cv2.VideoCapture(str(video));original_fps=reader.get(cv2.CAP_PROP_FPS);reader.release()
            if not np.isfinite(original_fps) or original_fps<=0:raise ValueError('Provide original_fps for a video with no FPS metadata')
        return self.parse_output(result,original_fps=original_fps,output_fps=fps if output_fps is None else output_fps)

    __call__=infer_monocular_motion_capture

    def save_pretrained(self,save_directory,**kwargs):
        import torch
        from motius.utils.torch_checkpoint import load_torch_checkpoint
        destination=Path(save_directory);root=self.bundle.artifact_root
        if root is None:raise ValueError('A complete capture export requires pipeline assets')
        destination.mkdir(parents=True,exist_ok=True)
        self.bundle.save_pretrained(destination,**kwargs)
        names=['camcalib_sa_biased_l2.ckpt','sam2_ckpts/keypoint_rcnn_5ad38f.pkl','sam2_ckpts/sam2_hiera_tiny.pt',
               'vitpose-h-coco_25.pth','deeplabv3_resnet50.pth','droidcalib.pth','metric_depth_vit_large_800k.pth']
        names=['data/pretrain/'+name for name in names]+['data/body_models/smplx2smpl.pkl','data/body_models/smplx2smpl_joints.npy']
        for name in names:
            source=root/name;target=destination/name
            if source.resolve()==target.resolve():continue
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        source=root/'data/pretrain/detector_native.pt'
        if source.is_file():
            target=destination/'data/pretrain/detector_native.pt'
            if source.resolve()!=target.resolve():shutil.copy2(source,target)
        source=root/'prompthmr_geometry.pt';target=destination/source.name
        if source.is_file():
            if source.resolve()!=target.resolve():shutil.copy2(source,target)
        else:
            state=load_torch_checkpoint(root/'checkpoint.ckpt')['state_dict']
            torch.save({key:value for key,value in state.items() if key.startswith(('smpl.','smplx.','smplx2smpl'))},target)
        # Persist current frontend parameters if the caller has updated an
        # instantiated frontend. Geometry and pretrained asset paths are owned
        # by this pipeline, never by the image/temporal neural bundle.
        if self._capture is not None:
            files=dict(camera=names[0],mask=names[2],pose=names[3],segmenter=names[4])
            for name,bundle in self._capture.frontends.items():
                state={key:value.detach().cpu() for key,value in bundle.model.state_dict().items()}
                if name=='detector':
                    torch.save(state,destination/'data/pretrain/detector_native.pt')
                else:
                    if name=='camera':state={'state_dict':{'model.'+key:value for key,value in state.items()}}
                    elif name=='mask':state={'model':state}
                    torch.save(state,destination/files[name])
        (destination/'prompthmr_pipeline.json').write_text(json.dumps(dict(options=self.options,source_checkpoint_sha256s=self.checkpoint_hashes()),indent=2)+'\n')
        return str(destination)
