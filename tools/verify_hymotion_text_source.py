"""Compare real Qwen3-8B/CLIP-L text processing against the pinned HY-Motion source."""
from pathlib import Path
import argparse,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def worker(args):
    import torch
    from verify_motionstreamer_source import digest
    if os.environ.get("MOTIUS_REFERENCE_HOST"):
        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream))
        from hymotion.network.text_encoders import text_encoder as module
        # Only resolve the original configuration paths to hash-verified local files.
        module.LLM_ENCODER_LAYOUT['qwen3']['module_path']=str(args.qwen)
        module.SENTENCE_EMB_LAYOUT['clipl']['module_path']=str(args.clip)
        network=module.HYTextModel(max_length_llm=64).cuda().eval()
        frontend=None
    else:
        from motius.models.hymotion_t2m.network.text_encoder import HYTextModel
        from motius.motion.conditioning.hymotion_text import HYTextProcessor
        config=dict(max_length_llm=64,llm_model_path=str(args.qwen),sentence_emb_model_path=str(args.clip))
        frontend=HYTextProcessor(**config)
        network=HYTextModel(**config).cuda().eval()
    state={}
    for key,value in network.state_dict().items():
        # Conditioning reads decoder hidden states; the source's unused
        # vocabulary projection is deliberately absent from the native tower.
        if key.startswith('llm_text_encoder.lm_head.'):
            continue
        if key.startswith('llm_text_encoder.model.'):
            key='llm_text_encoder.'+key[len('llm_text_encoder.model.'):]
        state[key]=digest(value)
    output={'weights':state,'cases':{},
            'clip_dtype':str(next(network.sentence_emb_text_encoder.parameters()).dtype),
            'qwen_dtype':str(next(network.llm_text_encoder.parameters()).dtype)}
    for index,texts in enumerate([['A person walks forward.'],['A person waves.',''],['   Walk   ','A person walks, turns and raises both arms.'],['word '*180]]):
        torch.manual_seed(71+index)
        values=network.encode(texts if frontend is None else frontend(texts))
        output['cases'][str(index)]=[value.cpu() for value in values]
    torch.save(output,args.output_dir/f'{args.worker}.pt')


def main():
    parser=argparse.ArgumentParser()
    for name in ['upstream','qwen','clip','output-dir']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--worker',choices=['upstream','native'])
    parser.add_argument('--reuse-results',action='store_true')
    args=parser.parse_args()
    for name in ['upstream','qwen','clip','output_dir']:setattr(args,name,getattr(args,name).resolve())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.worker:worker(args);return
    from verify_prompthmr_video_parity import compare_full as compare
    import torch
    report={'complete':False,'scope':'Independent original HY-Motion Qwen3-8B BF16 and CLIP-L FP32 text encoders, prompt/crop, dynamic sentence padding and all used neural hashes. Native decoder-only Qwen3 omits the original unused vocabulary projection; source model-prefix names are canonicalized. Four raw-text batches; excludes motion denoiser/training and MotionCanvas-specific frontend settings.',
            'upstream_commit':subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),'tests':[]}
    path=args.output_dir/'source_parity.json';path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ([] if args.reuse_results else ['upstream','native']):
        env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--upstream',str(args.upstream),'--qwen',str(args.qwen),'--clip',str(args.clip),'--output-dir',str(args.output_dir),'--worker',kind],cwd=ROOT,env=env,check=True)
    try:compare(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True;path.write_text(json.dumps(report,indent=2)+'\n');print('HY-Motion real text source comparisons exact')

if __name__=='__main__':main()
