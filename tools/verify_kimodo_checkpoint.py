"""Modified KIMODO neural/text weights must survive a portable offline cold export."""
from pathlib import Path
import argparse,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def execute(args):
    import torch
    from verify_prompthmr_video_parity import tensors
    from verify_motionstreamer_source import hashes
    from motius.pipelines.kimodo import KIMODOPipeline
    if os.environ.get("MOTIUS_REFERENCE_HOST"):
        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    os.environ.update(TEXT_ENCODER_MODE='local',TEXT_ENCODER='llm2vec',TOKENIZERS_PARALLELISM='false',
                      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    path=args.artifact if args.worker=='export' else args.output_dir/'exported'
    pipe=KIMODOPipeline.from_pretrained(str(path),device='cuda')
    model=pipe.runtime
    texts=['A person walks forward.']
    with torch.no_grad():
        initial=model.text_encoder(texts)[0]
        if args.worker=='export':
            norm=next(parameter for name,parameter in pipe.bundle.text_model.named_parameters() if name.endswith('model.norm.weight'))
            norm.mul_(1.125)
            last=list(pipe.bundle.network.denoiser.parameters())[-1]
            last.add_(0.03125)
        features=model.text_encoder(texts)[0]
        if args.worker=='export':assert not torch.equal(initial,features),'Text modification must affect the actual encoding'
        torch.manual_seed(417)
        generated=pipe.generate(texts,30,return_numpy=False,num_denoising_steps=3,post_processing=False)
    values=tensors({'features':features,'generated':generated,'neural':{'denoiser':hashes(pipe.bundle.network.denoiser),'text':hashes(pipe.bundle.text_model)}})
    torch.save(values,args.output_dir/f'{args.worker}.pt')
    if args.worker=='export':pipe.bundle.save_pretrained(str(args.output_dir/'exported'),copy_mode='hardlink')
    else:
        assert not any(name=='kimodo' or name.startswith('kimodo.') for name in sys.modules)
        config=json.loads((args.output_dir/'exported/kimodo_config.json').read_text())
        assert config['config']['packed_text_path']=='text_encoders'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--worker',choices=['export','cold'])
    args=parser.parse_args();args.artifact=args.artifact.resolve();args.output_dir=args.output_dir.resolve();args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.worker:execute(args);return
    report={'complete':False,'scope':'Actual modified KIMODO denoiser and 8B Llama/LoRA text tensors, full hashes and public raw-text generation through offline cold loading of a lean current-weight artifact. Migration/export evidence, not an independent original-source comparison.','tests':[]}
    path=args.output_dir/'source_parity.json';path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['export','cold']:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--artifact',str(args.artifact),'--output-dir',str(args.output_dir),'--worker',kind],cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT)},check=True)
    import torch
    from verify_prompthmr_video_parity import compare_full
    try:compare_full(torch.load(args.output_dir/'export.pt',weights_only=True),torch.load(args.output_dir/'cold.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True;path.write_text(json.dumps(report,indent=2)+'\n');print('KIMODO actual current neural/text weights and cold generation exact')

if __name__=='__main__':main()
