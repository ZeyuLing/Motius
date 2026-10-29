"""Isolated experiments; no changed-quality approximations or production defaults."""
import argparse,json,time,resource
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np,torch
from motius.pipelines.flowhmr.preprocess import FlowHMRVideoPreprocessor


class SubsetMHR(torch.nn.Module):
    @torch.no_grad()
    def __init__(self,inner,indices):
        super().__init__()
        self.inner=inner
        self.register_buffer('indices',indices)
        self.vertex_count=inner.character_torch.linear_blend_skinning.num_vertices
        for blend in [inner.character_torch.blend_shape,inner.face_expressions_model]:
            blend.shape_vectors=blend.shape_vectors[:,indices].contiguous()
        blend=inner.character_torch.blend_shape
        blend.base_shape=blend.base_shape[indices].contiguous()
        linear=inner.pose_correctives_model.pose_dirs_predictor._modules['2']
        linear.weight=linear.weight.reshape(self.vertex_count,3,-1)[indices].reshape(len(indices)*3,-1).contiguous()
        lbs=inner.character_torch.linear_blend_skinning
        remap=torch.full((self.vertex_count,),-1,device=indices.device,dtype=torch.long)
        remap[indices]=torch.arange(len(indices),device=indices.device)
        mask=remap[lbs.vert_indices_flattened]>=0
        lbs.skin_indices_flattened=lbs.skin_indices_flattened[mask].contiguous()
        lbs.skin_weights_flattened=lbs.skin_weights_flattened[mask].contiguous()
        lbs.vert_indices_flattened=remap[lbs.vert_indices_flattened[mask]].contiguous()
        lbs.num_vertices=len(indices)

    def forward(self,*args):
        vertices,skeleton=self.inner(*args)
        full=vertices.new_zeros(vertices.shape[0],self.vertex_count,3)
        full[:,self.indices]=vertices
        return full,skeleton


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--variant',choices=['baseline8','trim8','trim16','compile8','compact8','compact16','lean8','lean16','graph8','coalesced8','decoder_graph8'],required=True)
    p.add_argument('--frames',type=int,default=64)
    a=p.parse_args();root=a.root;out=root/'extreme'/a.variant;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(6*2**30/torch.cuda.get_device_properties(0).total_memory)
    start=time.perf_counter();paths=json.loads((root/'asset_paths.json').read_text())
    pre=FlowHMRVideoPreprocessor(device='cuda',sam_token_only=True,sam_fast=True,
        sam3d_ckpt=paths['preprocessor/sam/model.ckpt'],sam3d_config=paths['preprocessor/sam/model_config.yaml'],sam3d_mhr=paths['preprocessor/sam/assets/mhr_model.pt'])
    extractor=pre.build_sam_extractor();extractor._conditioning_cache=False
    model=extractor.estimator.model
    report=dict(variant=a.variant,load_seconds=time.perf_counter()-start,budget_gib=6,frames=a.frames,
        backbone_type=type(model.backbone).__name__,decoder_layers=len(model.decoder.layers),hand_decoder_layers=len(model.decoder_hand.layers),
        hand_mapping_nonzero_vertices=int((model.head_pose_hand.keypoint_mapping[:70,:18439].abs().sum(0)>0).sum()),
        body_mapping_nonzero_vertices=int((model.head_pose.keypoint_mapping[:70,:18439].abs().sum(0)>0).sum()))
    geometry_checks={}
    if a.variant.startswith(('lean','graph','coalesced','decoder_graph')):
        for head in [model.head_pose,model.head_pose_hand]:
            head.faces.data=head.faces.data.cpu()
            def no_vertices(module,args,output):
                output['pred_vertices']=None
                return output
            head.register_forward_hook(no_vertices)
        import types
        grid_cache={}
        def ray(self,batch):
            B,N,_,H,W=batch['img'].shape
            key=(H,W,batch['img'].device)
            if key not in grid_cache:
                grid_cache[key]=torch.stack(torch.meshgrid(torch.arange(H,device=key[2]),torch.arange(W,device=key[2]),indexing='xy'),2)[None,None]
            value=grid_cache[key].expand(B,N,-1,-1,-1)/batch['affine_trans'][:,:,None,None,[0,1],[0,1]]
            value=value-batch['affine_trans'][:,:,None,None,[0,1],[2,2]]/batch['affine_trans'][:,:,None,None,[0,1],[0,1]]
            value=value-batch['cam_int'][:,None,None,None,[0,1],[2,2]]
            value=value/batch['cam_int'][:,None,None,None,[0,1],[0,1]]
            return value.permute(0,1,4,2,3).to(batch['img'].dtype)
        model.get_ray_condition=types.MethodType(ray,model)
        def no_mask(self,batch,embeddings):
            return self.prompt_encoder.get_mask_embeddings(None,embeddings.shape[0],embeddings.shape[2:])[1].to(embeddings)
        model._get_mask_prompt=types.MethodType(no_mask,model)
        def cached(module):
            original=module.forward;values={}
            def forward(*args,**kwargs):
                tensor=args[0] if args else kwargs['keypoints']
                key=(tuple(tensor.shape),tensor.dtype,tensor.device)
                if key not in values:values[key]=original(*args,**kwargs)
                return values[key]
            module.forward=forward
        for module in [model.prompt_encoder,model.prev_to_token_mhr,model.prev_to_token_mhr_hand]:cached(module)
        original_pe=model.prompt_encoder.get_dense_pe;positions={}
        def dense_pe(size):
            if tuple(size) not in positions:positions[tuple(size)]=original_pe(size)
            return positions[tuple(size)]
        model.prompt_encoder.get_dense_pe=dense_pe
    if a.variant.startswith(('trim','compact','lean','graph','coalesced','decoder_graph')):

        for decoder in [model.decoder,model.decoder_hand]:
            original=decoder.forward;last=len(decoder.layers)-1
            def wrapped(*args,_original=original,_last=last,**kwargs):
                callback=kwargs['token_to_pose_output_fn']
                def prediction(tokens,prev_pose_output,layer_idx):
                    return {} if layer_idx==_last else callback(tokens,prev_pose_output,layer_idx)
                kwargs['token_to_pose_output_fn']=prediction
                return _original(*args,**kwargs)
            decoder.forward=wrapped
    if a.variant in ('graph8','coalesced8','decoder_graph8'):
        for head in [model.head_pose,model.head_pose_hand]:
            sparse=head.mhr.pose_correctives_model.pose_dirs_predictor._modules['0']
            matrix=torch.sparse_coo_tensor(sparse.sparse_indices,sparse.sparse_weight,sparse.sparse_shape).coalesce()
            sparse.sparse_indices=matrix.indices()
            sparse.sparse_weight=matrix.values()
            graph=sparse.forward.graph
            for node in graph.nodes():
                if node.kind()=='aten::sparse_coo_tensor':
                    with graph.insert_point_guard(node):flag=graph.insertConstant(True)
                    node.replaceInput(7,flag)
            torch._C._jit_pass_lint(graph)
    if a.variant=='graph8':
        model.backbone.requires_grad_(False)
        original_backbone=model.backbone.forward
        graphs={};pool=torch.cuda.graph_pool_handle()
        def backbone(*args,**kwargs):
            tensor=args[0];key=tuple(tensor.shape)
            if tensor.shape[0] not in (8,16):return original_backbone(*args,**kwargs)
            if key not in graphs:
                static=tensor.clone();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):original_backbone(static,**kwargs)
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,pool=pool,stream=stream):output=original_backbone(static,**kwargs)
                torch.cuda.current_stream().wait_stream(stream)
                graphs[key]=(static,graph,output)
            static,graph,output=graphs[key];static.copy_(tensor);graph.replay();return output
        model.backbone.forward=backbone
    if a.variant=='decoder_graph8':
        from motius.pipelines.flowhmr.features import _TokensCaptured
        graph_records={};pool=torch.cuda.graph_pool_handle()
        def clone(value):
            if isinstance(value,torch.Tensor):return value.clone()
            if isinstance(value,dict):return {k:clone(v) for k,v in value.items()}
            if isinstance(value,tuple):return tuple(clone(v) for v in value)
            return value
        def copy_values(target,source):
            if isinstance(source,torch.Tensor):target.copy_(source)
            elif isinstance(source,dict):
                for k in target:copy_values(target[k],source[k])
            elif isinstance(source,tuple):
                for a,b in zip(target,source):copy_values(a,b)
        def configure(original,label):
            def call(*args,**kwargs):
                batch=len(args[0])
                if batch not in ((8,) if label=='body' else (16,)):
                    return original(*args,**kwargs)
                if label not in graph_records:
                    active=extractor._capture_active;saved_body=list(extractor._body_tokens);saved_hand=list(extractor._hand_tokens)
                    extractor._capture_active=False
                    try:
                        static_args,static_kwargs=clone(args),clone(kwargs)
                        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(3):
                                extractor._body_tokens.clear();extractor._hand_tokens.clear()
                                original(*static_args,**static_kwargs)
                        torch.cuda.current_stream().wait_stream(stream)
                        extractor._body_tokens.clear();extractor._hand_tokens.clear()
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph,pool=pool,stream=stream):result=original(*static_args,**static_kwargs)
                        torch.cuda.current_stream().wait_stream(stream)
                        tokens=(extractor._body_tokens if label=='body' else extractor._hand_tokens)[-1]
                        graph_records[label]=(static_args,static_kwargs,graph,result,tokens)
                    finally:
                        extractor._capture_active=active;extractor._body_tokens[:]=saved_body;extractor._hand_tokens[:]=saved_hand
                static_args,static_kwargs,graph,result,tokens=graph_records[label]
                copy_values(static_args,args);copy_values(static_kwargs,kwargs);graph.replay()
                if label=='body':extractor._body_tokens.append(tokens.detach())
                else:
                    extractor._hand_tokens.extend(tokens.detach().split(extractor._body_tokens[0].shape[0],0))
                    raise _TokensCaptured
                return result
            return call
        model.forward_decoder=configure(model.forward_decoder,'body')
        model.forward_decoder_hand=configure(model.forward_decoder_hand,'hand')
    if a.variant=='compile8':
        model.backbone=torch.compile(model.backbone,fullgraph=True,dynamic=True,
            options={'emulate_precision_casts':True,'triton.cudagraphs':False})
    inputs=np.load(root/'vggt-full/fast/inputs.npz');boxes=inputs['bbox'][:a.frames];cameras=inputs['camera_K'][:a.frames]
    cap=cv2.VideoCapture(str(root/'fast-sam-full/upstream/video/input_30fps.mp4'));images=[]
    try:
        for _ in range(a.frames):
            ok,frame=cap.read()
            if not ok:raise ValueError('Video ended early')
            images.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    finally:cap.release()
    if a.variant.startswith('compact'):
        saved={}
        class Observe(torch.nn.Module):
            def __init__(self,inner,label):
                super().__init__();self.inner=inner;self.label=label
            def forward(self,*args):
                output=self.inner(*args)
                if self.label not in saved:
                    saved[self.label]=(tuple(arg.detach().clone() for arg in args),tuple(v.detach().clone() for v in output))
                return output
        for label,head in [('body',model.head_pose),('hand',model.head_pose_hand)]:
            head.mhr=Observe(head.mhr,label)
        extractor.extract_frames(images[:8],boxes[:8],cameras[:8])
        for label,head in [('body',model.head_pose),('hand',model.head_pose_hand)]:
            indices=(head.keypoint_mapping[:70,:18439].abs().sum(0)>0).nonzero()[:,0]
            head.mhr=SubsetMHR(head.mhr.inner,indices)
            with torch.no_grad():new=head.mhr(*saved[label][0])
            ref=saved[label][1]
            geometry_checks[label]={'kept_vertices':len(indices),'vertex_max_abs_native':float((new[0][:,indices]-ref[0][:,indices]).abs().max()),'skeleton_bitwise_equal':bool(torch.equal(new[1],ref[1]))}
    size=16 if a.variant.endswith('16') else 8
    started=time.perf_counter()
    extractor.extract_frames(images[:size],boxes[:size],cameras[:size])
    torch.cuda.synchronize();report['first_batch_seconds']=time.perf_counter()-started
    events=defaultdict(list)
    for name,module in [('backbone',model.backbone),('body_decoder',model.decoder),('hand_decoder',model.decoder_hand),('body_mhr',model.head_pose),('hand_mhr',model.head_pose_hand)]:
        active=[]
        def before(module,args,_active=active):
            event=torch.cuda.Event(enable_timing=True);event.record();_active.append(event)
        def after(module,args,output,_active=active,_name=name):
            event=torch.cuda.Event(enable_timing=True);event.record();events[_name].append((_active.pop(),event))
        module.register_forward_pre_hook(before)
        module.register_forward_hook(after,always_call=True)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter();tokens=[]
    for index in range(0,a.frames,size):
        tokens.append(extractor.extract_frames(images[index:index+size],boxes[index:index+size],cameras[index:index+size]))
    torch.cuda.synchronize();features=torch.cat(tokens).numpy()
    report.update(geometry_checks=geometry_checks,seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        gpu_module_seconds={name:sum(first.elapsed_time(last) for first,last in pairs)/1000 for name,pairs in events.items()},
        host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    reference=np.load(root/'vggt-full/fast/output.npz')['feature'][:a.frames]
    x=reference.reshape(-1,3,1024).astype(float);y=features.reshape(-1,3,1024).astype(float)
    cosine=(x*y).sum(-1)/(np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1))
    report.update(bitwise_equal_to_previous=bool(np.array_equal(reference,features)),token_cosine_min=cosine.min(0).tolist(),max_abs=float(abs(reference-features).max()))
    np.savez(out/'features.npz',feature=features);(out/'report.json').write_text(json.dumps(report,indent=2));print('EXTREME_RESULT',json.dumps(report),flush=True)

if __name__=='__main__':main()
