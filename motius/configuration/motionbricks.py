"""Whitelisted MotionBricks checkpoint metadata and representation construction."""
from pathlib import Path
from omegaconf import OmegaConf


def checkpoint_root(path):
    root=Path(path).expanduser().resolve()
    if (root/'motionbricks_checkpoint').is_dir():root=root/'motionbricks_checkpoint'
    return root


def source_config(root,stage='pose'):
    return OmegaConf.load(checkpoint_root(root)/('motionbricks_'+stage)/'version_1/hparams.yaml')


def build_representation(root,stage='pose'):
    from motius.motion.representation.motionbricks.skeletons.g1 import G1Skeleton34
    from motius.motion.representation.motionbricks.utils.stats import Stats
    from motius.motion.representation.motionbricks.motion_reps.dual_root_global_joints import GlobalRootGlobalJoints
    root=checkpoint_root(root);cfg=source_config(root,stage)
    if cfg.skeleton.name!='g1skel34':raise ValueError('This release requires G1Skeleton34')
    folder=root/('motionbricks_'+stage)/'version_1'
    skeleton=G1Skeleton34(folder=str(folder/'skeleton'),name=cfg.skeleton.name,t_pose=cfg.skeleton.get('t_pose','capture'))
    stats=Stats(folder=str(folder/'stats/motion'))
    return GlobalRootGlobalJoints(fps=cfg.fps,skeleton=skeleton,name=cfg.motion_rep.name,stats=stats)


def network_options(root):
    """Read legacy released YAML as data; never instantiate arbitrary targets."""
    import torch
    from motius.motion.conditioning.motionbricks import extract_feature_from_motion_rep
    rep=build_representation(root);local=rep.dual_rep.local_motion_rep;global_rep=rep.dual_rep.global_motion_rep
    dimension=lambda representation,feature:extract_feature_from_motion_rep(torch.zeros(1,1,len(representation.indices["all"])),representation,feature).shape[-1]
    pose_cfg=source_config(root,'pose');root_cfg=source_config(root,'root')
    pose=OmegaConf.to_container(pose_cfg.model.backbone_network.args,resolve=True)
    root_options=OmegaConf.to_container(root_cfg.model.backbone_network.args,resolve=True)
    pose['local_root_dim']=dimension(local if pose['cond_root_feature_is_from_motion_rep']=='local' else global_rep,pose['cond_root_feature'])
    pose['local_pose_dim']=dimension(local,pose['local_pose_feature'])
    for key,representation in [('local_root',local),('global_root',global_rep),('local_pose',local)]:
        root_options[key+'_dim']=dimension(representation,root_options[key+'_feature'])
    tokenizer=OmegaConf.to_container(pose_cfg.model.pose_vqvae_network,resolve=True)
    for key in ['_target_','motion_rep']:tokenizer.pop(key,None)
    features=tokenizer['feature_mode'];token_rep=local if pose_cfg.model.pose_vqvae_motion_rep=='local' else global_rep
    for key,feature in zip(['encoder_state_dim','decoder_state_dim','decoder_target_cond_dim','decoder_external_cond_dim'],features):
        tokenizer[key]=dimension(token_rep,feature)
    return dict(pose=pose,root=root_options,tokenizer=tokenizer,
                pose_training=OmegaConf.to_container(pose_cfg.model.args,resolve=True),
                root_training=OmegaConf.to_container(root_cfg.model.args,resolve=True))


from dataclasses import dataclass
@dataclass(frozen=True)
class MotionBricksCheckpointLayout:
    """Official MotionBricks checkpoint files expected by the runtime."""
    root: Path
    clip: Path
    vqvae: Path
    pose: Path
    root_model: Path

    @classmethod
    def from_root(cls, root: Path) -> 'MotionBricksCheckpointLayout':
        return cls(root=root, clip=root / 'G1-clip.ckpt', vqvae=root / 'motionbricks_vqvae/version_1/checkpoints/model-step=2000000.ckpt', pose=root / 'motionbricks_pose/version_1/checkpoints/model-step=2000000.ckpt', root_model=root / 'motionbricks_root/version_1/checkpoints/model-step=2000000.ckpt')

    def missing(self) -> list[Path]:
        return [path for path in (self.clip, self.vqvae, self.pose, self.root_model) if not path.is_file()]

    def lfs_pointers(self) -> list[Path]:
        pointers: list[Path] = []
        for path in (self.clip, self.vqvae, self.pose, self.root_model):
            if not path.is_file() or path.stat().st_size > 4096:
                continue
            try:
                if path.read_text(errors='ignore').startswith('version https://git-lfs.github.com/spec/v1'):
                    pointers.append(path)
            except OSError:
                pass
        return pointers

def validate_checkpoint_layout(root):
    layout=MotionBricksCheckpointLayout.from_root(checkpoint_root(root))
    missing=layout.missing();pointers=layout.lfs_pointers()
    if missing or pointers:
        raise FileNotFoundError('MotionBricks checkpoints: '+('missing: '+', '.join(map(str,missing)) if missing else '')+(' git-lfs pointers: '+', '.join(map(str,pointers)) if pointers else ''))
    return layout
