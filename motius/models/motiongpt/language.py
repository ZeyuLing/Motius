"""Atomic MotionGPT language network, preserving released checkpoint keys."""
from torch import nn
from transformers import T5ForConditionalGeneration


class MotionGPTLanguageNetwork(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.language_model=T5ForConditionalGeneration(config)

    def forward(self,*args,**kwargs):
        return self.language_model(*args,**kwargs)
