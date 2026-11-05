"""Native MotionBricks network, primitive and robot-geometry composition."""
from pathlib import Path
import json,shutil
from typing import Any
import torch
from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES
from motius.configuration.motionbricks import checkpoint_root
from .navigation import build_navigation,rollout_navigation


@PIPELINES.register_module()
class MotionBricksPipeline(BasePipeline):
    BUNDLE_CLS='motius.models.motionbricks.MotionBricksBundle'
    fps=30
    representation='motionbricks_g1_414'

    def __init__(self,bundle,*,controller=None,clips=None,**runtime_options):
        super().__init__(bundle)
        root=bundle.artifact_root
        config={}
        if root is not None:
            file=root/'motionbricks_pipeline.json'
            if file.exists():config=json.loads(file.read_text())
            elif (root/'motionbricks_config.json').exists():config=json.loads((root/'motionbricks_config.json').read_text()).get('config',{})
        self.controller=controller or config.get('controller','wasd')
        self.clips=clips or config.get('clips','G1')
        self.runtime_options={**config.get('runtime_kwargs',{}),**runtime_options}
        self._agent=None

    def validate_checkpoints(self):
        root=checkpoint_root(self.bundle.artifact_root)
        if not (root/'G1-clip.ckpt').is_file():raise FileNotFoundError(root/'G1-clip.ckpt')

    def build_demo_agent(self,*,seed=None,controller=None,**overrides:Any):
        self.validate_checkpoints()
        options={**self.runtime_options,**overrides}
        self._agent=build_navigation(self.bundle,controller=controller or self.controller,clips=self.clips,seed=seed,**options)
        return self._agent

    @torch.inference_mode()
    def rollout(self,steps=120,*,controller=None,seed=None,**overrides):
        agent=self.build_demo_agent(controller=controller,seed=seed,**overrides)
        return rollout_navigation(agent,steps)

    infer_g1_realtime_navigation=rollout
    infer_g1_qpos_generation=rollout

    def infer_t2m(self,*args,**kwargs):raise NotImplementedError('Use the G1 navigation or qpos task API')

    def save_pretrained(self,save_directory,**kwargs):
        destination=Path(save_directory);self.bundle.save_pretrained(destination,**kwargs)
        root=checkpoint_root(self.bundle.artifact_root)
        for stage in ['pose','root','vqvae']:
            source=root/('motionbricks_'+stage)/'version_1';target=destination/('motionbricks_'+stage)/'version_1'
            for name in ['hparams.yaml','skeleton','stats']:
                file=source/name
                if not file.exists():raise FileNotFoundError(file)
                out=target/name
                if file.resolve()==out.resolve():continue
                out.parent.mkdir(parents=True,exist_ok=True)
                if file.is_dir():shutil.copytree(file,out,dirs_exist_ok=True)
                else:shutil.copy2(file,out)
        source=root/'G1-clip.ckpt';target=destination/source.name
        if source.resolve()!=target.resolve():shutil.copy2(source,target)
        (destination/'motionbricks_pipeline.json').write_text(json.dumps(dict(controller=self.controller,clips=self.clips,runtime_kwargs=self.runtime_options),indent=2)+'\n')
        return str(destination)
