"""Compare the port against an extracted, trusted original MoGenDiT archive.

Run: python tools/verify_mogendit_parity.py --source /path/to/MoGenDiT
Uses small CPU networks; no pretrained weights are needed.
"""
import argparse
from pathlib import Path
import sys
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
args = parser.parse_args()
if not (args.source / 'model/more_diff.py').is_file():
    parser.error('--source must be the extracted MoGenDiT directory')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(args.source.resolve()))
from model.more_diff import MoreDiff as Original
from EasyDiffusion import GaussianDiffusion as OriginalDiffusion
from motion_process.motion_representation import OccamMotionRep as OriginalRep
from motius.models.mogendit import MoreDiff
from motius.motion.representation.mogendit.motion_representation import OccamMotionRep
from motius.diffusion.mogendit.base_diffusion import GaussianDiffusion

torch.set_num_threads(1)
torch.manual_seed(83)
cfg=dict(d_motion=201,d_model=16,d_cond=66,n_head=2,n_stack=2,dropout=0.,window_size=4)
a=Original(**cfg).eval(); b=MoreDiff(**cfg).eval()
# Exercise attention/conditioning instead of only the zero-initialized output.
with torch.no_grad():
 for p in a.parameters():
  if p.requires_grad: p.normal_(0,0.1)
b.load_state_dict(a.state_dict(),strict=True)
x=torch.randn(2,7,201);cond=torch.randn(2,1,66);mask=torch.randint(0,2,x.shape).float();pad=torch.tensor([[False]*7,[False]*5+[True]*2]);t=torch.tensor([10,900])
ya=a(a.wrap_inputs(x,cond,mask,pad),t);yb=b(b.wrap_inputs(x,cond,mask,pad),t)
torch.testing.assert_close(ya,yb,rtol=0,atol=0)
print('Network forward: bit-exact; state-dict keys:',len(a.state_dict()))
for cls,label in [(OriginalDiffusion,'original'),(GaussianDiffusion,'ported')]:
 torch.manual_seed(17)
 out=cls().denoise(b.wrap_inputs(x.clone(),cond,mask,pad), b, num_timesteps=3,eta=1,device=torch.device('cpu'),mask=mask.bool(),imputation_mode='skip_last')
 if label=='original': ref=out
 else: torch.testing.assert_close(ref,out,rtol=0,atol=0)
print('Seeded diffusion: bit-exact')
ra=OriginalRep();rb=OccamMotionRep()
pose=torch.eye(3).expand(7,22,3,3).clone(); joints=torch.randn(7,22,3); joints[:,0]=0;trans=torch.randn(7,3)
xa=ra.encode(pose,joints,trans);xb=rb.encode(pose,joints,trans)
torch.testing.assert_close(xa,xb,rtol=0,atol=0)
torch.testing.assert_close(ra.normalization(xa,height_reset=True),rb.normalization(xb,height_reset=True),rtol=0,atol=0)
print('Occam encoding/normalization: bit-exact')
