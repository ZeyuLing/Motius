"""Compare losses/gradients with methods extracted from the supplied original trainer.

Only the original objective methods are compiled: missing Aplus trainer/UI
imports are not required, and the original optimizer loop is not executed.
"""
import argparse
import ast
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motius.models.mogendit import MoGenDiTBundle
from motius.trainers.mogendit import MoGenDiTTrainer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args=parser.parse_args()
    path=args.source/'trainer/my_trainer.py'
    if not path.is_file():
        parser.error("--source must include the original trainer/my_trainer.py")
    sys.path.insert(0, str(args.source.resolve()))
    from motion_process.motion_representation import OccamMotionRep as OriginalRep
    original_rep = OriginalRep()
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MoGenDitDistributedTrainer')
    funcs={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
    forward=funcs['forward_backward']
    stop=next(i for i,n in enumerate(forward.body) if isinstance(n,ast.Expr)
              and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
              and n.value.func.attr=='backward')
    forward.body=forward.body[:stop]+[ast.Return(value=ast.Name(id='loss',ctx=ast.Load()))]
    weighted=funcs['weighted_masked_loss']
    module=ast.fix_missing_locations(ast.Module(body=[weighted,forward],type_ignores=[]))
    scope={'torch':torch}
    exec(compile(module,str(path),'exec'),scope)
    torch.set_num_threads(1)
    torch.manual_seed(15)
    bundle=MoGenDiTBundle(network_config=dict(d_model=16,n_head=2,n_stack=1))
    trainer=MoGenDiTTrainer(bundle, consis_start_step=3,l1_weight_x0=.7,l2_weight_x0=.3,
                            l1_weight_consis=.6,l2_weight_consis=.4)
    clean=torch.randn(2,6,201)
    lengths=torch.tensor([6,4]);weights=torch.tensor([.75,1.25])
    valid=trainer.mask_scheduler.get_length_mask_bool(clean,lengths)
    prediction=torch.randn_like(clean)*.2
    for step in (0,3,4):
        old_pred=prediction.clone().requires_grad_()
        reference=SimpleNamespace(
            optimizer=SimpleNamespace(zero_grad=lambda:None), model=lambda *_:old_pred,
            motion_rep=original_rep,iter=step,
            args=SimpleNamespace(consis_loss=True,consis_start_step=3,l1_weight_x0=.7,l2_weight_x0=.3,
                                 l1_weight_consis=.6,l2_weight_consis=.4))
        reference.weighted_masked_loss=MethodType(scope['weighted_masked_loss'],reference)
        original=scope['forward_backward'](reference,({},None,clean,None,weights,valid,lengths)).squeeze()
        original.backward()
        new_pred=prediction.clone().requires_grad_()
        migrated,_=trainer.compute_losses(new_pred,clean,lengths,weights,step=step)
        migrated.backward()
        torch.testing.assert_close(original,migrated,rtol=0,atol=0)
        torch.testing.assert_close(old_pred.grad,new_pred.grad,rtol=1e-6,atol=2e-8)
        error = (old_pred.grad-new_pred.grad).abs().max().item()
        print(f'step={step}: loss bit-exact; max gradient difference={error:.3g}')


if __name__=='__main__':
    main()
