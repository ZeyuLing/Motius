"""Audit exact GEM-X objectives and distinguish original backward nondeterminism.

First run verify_gem_smpl_training_parity.py --variant gem_x. Its complete
zero-tolerance report remains false if any gradients differ. Run the original
worker again into a separate output directory, then pass both directories here.
This reports objective equivalence separately; it never certifies the gradients.
"""
import argparse,json
from pathlib import Path
import torch


def audit(output_dir,repeat_dir):
    original=torch.load(output_dir/'upstream.pt',map_location='cpu',weights_only=True)
    native=torch.load(output_dir/'native.pt',map_location='cpu',weights_only=True)
    repeated=torch.load(repeat_dir/'upstream.pt',map_location='cpu',weights_only=True)
    metadata=json.loads((output_dir/'training_parity.json').read_text())
    report={key:value for key,value in metadata.items() if key!='tests'}
    report.update(complete=False,comparison='Original GEM-X raw augmentation and full released weighted regression/joint objectives versus native Trainer',
        gradient_status='Uncertified: the original repeated backward and source/native gradients are recorded separately; no relaxed gradient tolerance is accepted.',tests=[],gradient_repeats={})
    path=output_dir/'objective_parity.json';path.write_text(json.dumps(report,indent=2)+'\n')
    for stage,values in original.items():
        assert values.keys()==native[stage].keys()==repeated[stage].keys(),stage
        for key,value in values.items():
            if isinstance(value,torch.Tensor):
                torch.testing.assert_close(value,native[stage][key],rtol=0,atol=0,msg=stage+'/'+key)
                torch.testing.assert_close(value,repeated[stage][key],rtol=0,atol=0,msg='repeat/'+stage+'/'+key)
                report['tests'].append(dict(tensor='/'+stage+'/'+key,shape=list(value.shape),bitwise_equal=True,max_abs_error=0.0))
        grads=values['gradients'];assert grads.keys()==native[stage]['gradients'].keys()==repeated[stage]['gradients'].keys()
        rows=[]
        for name,value in grads.items():
            candidate=native[stage]['gradients'][name];second=repeated[stage]['gradients'][name]
            assert torch.isfinite(value).all() and torch.isfinite(candidate).all() and torch.isfinite(second).all(),name
            rows.append(dict(parameter=name,source_native_max_abs=float((value-candidate).abs().max()),source_repeat_max_abs=float((value-second).abs().max()),bitwise_equal_native=torch.equal(value,candidate),bitwise_equal_repeat=torch.equal(value,second)))
        report['gradient_repeats'][stage]=rows
    report['complete']=True;path.write_text(json.dumps(report,indent=2)+'\n')
    print('Exact objective comparisons:',len(report['tests']))
    for stage,rows in report['gradient_repeats'].items():print(stage,'original repeated gradients differing:',sum(not row['bitwise_equal_repeat'] for row in rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--repeat-dir',type=Path,required=True)
    args=parser.parse_args();audit(args.output_dir,args.repeat_dir)
