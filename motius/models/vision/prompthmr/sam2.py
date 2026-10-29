"""Atomic neural calls for the released SAM 2 temporal architecture."""
from sam2.modeling.sam2_base import SAM2Base

class PromptSAM2Network(SAM2Base):
    def forward(self,*args,phase='image',**kwargs):
        if phase=='image':return self.forward_image(*args,**kwargs)
        if phase=='track':return self.track_step(*args,**kwargs)
        if phase=='memory':return self._encode_new_memory(*args,**kwargs)
        raise ValueError(phase)
