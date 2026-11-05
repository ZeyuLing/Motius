"""Probe installed SDPA backends at VGGT's actual all-frame key length."""
import argparse,json,time
from pathlib import Path
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(0)
torch.cuda.set_per_process_memory_fraction(6*2**30/torch.cuda.get_device_properties(0).total_memory)
q=torch.randn(1,16,4096,64,device='cuda',dtype=torch.bfloat16)
k=torch.randn(1,16,444750,64,device='cuda',dtype=torch.bfloat16)
v=torch.randn_like(k);rows=[];reference=None
for name in ['FLASH_ATTENTION','EFFICIENT_ATTENTION','CUDNN_ATTENTION']:
 try:
  with sdpa_kernel(getattr(SDPBackend,name)):
   for _ in range(2):y=torch.nn.functional.scaled_dot_product_attention(q,k,v)
   torch.cuda.synchronize();start=time.perf_counter()
   for _ in range(5):y=torch.nn.functional.scaled_dot_product_attention(q,k,v)
   torch.cuda.synchronize();seconds=(time.perf_counter()-start)/5
   if reference is None:reference=y.clone()
   rows.append(dict(backend=name,seconds=seconds,max_abs_vs_flash=float((y-reference).abs().max())))
 except Exception as e:rows.append(dict(backend=name,error=str(e)))
 print(json.dumps(rows[-1]),flush=True)
(a.root/'vggt-schedules/sdpa-backends.json').write_text(json.dumps(rows,indent=2))
