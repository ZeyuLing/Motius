"""Installed exact attention comparison at the full 750-frame key length."""
import argparse,json,sys,time
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
sys.path.insert(0,str(root/'fa3-deps'))
from flash_attn_3 import flash_attn_interface
q=torch.randn(1,16,4096,64,device='cuda',dtype=torch.bfloat16)
k=torch.randn(1,16,444750,64,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
torch.cuda.set_per_process_memory_fraction(6*2**30/torch.cuda.get_device_properties(0).total_memory)
rows=[];reference=None
for name in ['cudnn','fa3']:
 def run():
  if name=='cudnn':
   with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):return torch.nn.functional.scaled_dot_product_attention(q,k,v)
  result=flash_attn_interface.flash_attn_func(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),causal=False)
  return (result[0] if isinstance(result,tuple) else result).transpose(1,2)
 for _ in range(2):y=run()
 torch.cuda.synchronize();start=time.perf_counter()
 for _ in range(5):y=run()
 torch.cuda.synchronize();seconds=(time.perf_counter()-start)/5
 if reference is None:reference=y.clone()
 rows.append(dict(backend=name,seconds=seconds,max_abs=float((y-reference).abs().max()),peak_allocated_bytes=torch.cuda.max_memory_allocated()))
 print('ATTENTION_RESULT',json.dumps(rows[-1]),flush=True)
(root/'extreme/attention.json').write_text(json.dumps(rows,indent=2))
