"""Resolved GEM architecture data; runtime construction never imports Hydra targets."""
import copy,json
from pathlib import Path
from omegaconf import OmegaConf


def load_gem_config(variant,overrides=None):
    if variant not in {'gem_smpl','gem_x'}:raise ValueError(variant)
    data=json.loads((Path(__file__).parent/(variant+'.json')).read_text())
    config=OmegaConf.create(data)
    if overrides is not None:config=OmegaConf.merge(config,OmegaConf.create(overrides))
    return config
