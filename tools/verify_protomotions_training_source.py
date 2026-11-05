"""Original PPO/L2C2/AMP objectives and optimizer steps versus native Trainer."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
from verify_prompthmr_video_parity import tensors,compare_full


def canonical(values):
    prefixes={'_actor.mu.':'actor.mean.','_actor.':'actor.','_critic.':'critic.',
        '_discriminator.models.0.':'discriminator.','_disc_critic.models.0.':'discriminator_critic.'}
    result={}
    for key,value in values.items():
        for source,target in prefixes.items():
            if key.startswith(source):
                result[(target+key[len(source):]).replace('running_obs_norm.','')]=value;break
        else:raise ValueError('Unexpected original neural key: '+key)
    return result


def worker(args):
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(1831)
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream))
        from lightning.fabric import Fabric
        from protomotions.agents.amp.model import AMPModel
        from protomotions.agents.ppo.agent import PPO
        from protomotions.agents.amp.component import AMPTrainingComponent
        from protomotions.agents.utils.normalization import materialize_lazy_running_stats_from_state_dict
        config=torch.load(args.artifact/'resolved_configs.pt',map_location='cpu',weights_only=False)['agent']
        config.amp_parameters.discriminator_batch_size=4
        model=AMPModel(config.model)
        weights=torch.load(args.artifact/'last.ckpt',map_location='cpu',weights_only=False)['model']
        materialize_lazy_running_stats_from_state_dict(model,weights);model.load_state_dict(weights,strict=True);model.cuda().train()
        fabric=Fabric(accelerator='cuda',devices=1,precision='32-true')
        # The loss methods require no simulator. Bind their actual networks and
        # actual Fabric optimizers to source objects without constructing an env.
        agent=object.__new__(PPO);agent.config=config;agent.device=torch.device('cuda')
        agent.fabric=fabric;agent.num_envs=4;agent.e_clip=config.e_clip;agent.current_epoch=0
        component=object.__new__(AMPTrainingComponent);component.agent=agent
        modules=dict(actor=model._actor,critic=model._critic,discriminator=model._discriminator,
                     discriminator_critic=model._disc_critic)
        optimizers={}
        for role,module in modules.items():
            key='disc_critic' if role=='discriminator_critic' else role
            options=getattr(config.model,key+'_optimizer')
            optimizer=torch.optim.Adam(module.parameters(),lr=options.lr,betas=options.betas,eps=options.eps,weight_decay=options.weight_decay)
            wrapped,optimizer=fabric.setup(module,optimizer)
            optimizers[role]=optimizer
            if role in {'actor','critic'}:setattr(agent,role,wrapped)
            else:setattr(component,'disc_critic' if role=='discriminator_critic' else role,wrapped)
    else:
        from motius.models.protomotions import ProtoMotionsPolicyBundle
        from motius.trainers.protomotions.trainer import ProtoMotionsPPOTrainer
        model=ProtoMotionsPolicyBundle.from_original_checkpoint(args.artifact/'last.ckpt',device='cuda').train()
        trainer=ProtoMotionsPPOTrainer(model,discriminator_batch_size=4)
        modules={role:getattr(model,role) for role in ['actor','critic','discriminator','discriminator_critic']}
        optimizers={role:torch.optim.Adam(module.parameters(),lr=2e-5 if role=='actor' else 1e-4,
            betas=(.95,.99) if role in {'actor','critic'} else (.9,.999),eps=1e-8,weight_decay=0.) for role,module in modules.items()}
        trainer.set_optimizers(optimizers)
    dimensions=dict(noisy_reduced_coords_obs=64,noisy_mimic_reduced_coords_target_poses=256,
        clean_reduced_coords_obs=64,clean_mimic_reduced_coords_target_poses=256,
        max_coords_obs=493,mimic_max_coords_target_poses=3168,historical_max_coords_obs=3451,
        historical_previous_processed_actions=29)
    results={}
    for iteration in range(3):
        torch.manual_seed(1832+iteration)
        batch={name:torch.randn(4,size,device='cuda')*.17 for name,size in dimensions.items()}
        batch.update(action=torch.randn(4,29,device='cuda')*.1,neglogp=torch.randn(4,device='cuda')*.1,
            advantages=torch.randn(4,device='cuda'),value=torch.randn(4,1,device='cuda')*.1,
            returns=torch.randn(4,device='cuda')*.1,disc_value=torch.randn(4,1,device='cuda')*.1,
            disc_returns=torch.randn(4,device='cuda')*.1)
        for prefix in ['expert','agent','replay']:
            batch[prefix+'_historical_max_coords_obs']=torch.randn(4,3451,device='cuda')*.17
        if iteration==2:
            # Exercise an ordinary nonzero discriminator-value update as well
            # as the source excessive-gradient guard triggered by other cases.
            with torch.no_grad():
                if args.worker=='upstream':
                    from tensordict import TensorDict
                    component.disc_critic.eval()
                    prediction=component.disc_critic(TensorDict(batch,batch_size=[4]))['disc_value'].clone()
                    component.disc_critic.train()
                else:
                    model.discriminator_critic.eval();prediction=model.predict_discriminator_value(batch).clone()
                    model.discriminator_critic.train()
            batch['disc_value']=prediction+.01
            batch['disc_returns']=prediction.squeeze(-1)+torch.linspace(-.025,.025,4,device='cuda')
        for index,role in enumerate(['actor','critic','discriminator_critic','discriminator']):
            torch.manual_seed(1900+iteration*4+index)
            if args.worker=='upstream':
                if role=='actor':
                    loss,logs=agent.actor_step(batch)
                    metrics={name:logs[key] for name,key in [('ppo','actor/ppo_loss'),('bounds','actor/bounds_loss'),('extra','actor/extra_loss'),('entropy','actor/entropy_loss')]}
                elif role=='critic':loss,logs=agent.critic_step(batch);metrics={}
                elif role=='discriminator_critic':loss,logs=component.disc_critic_step(batch);metrics={}
                else:
                    loss,logs=component.discriminator_step(batch)
                    metrics={name:logs[key] for name,key in [('classification','discriminator/class_loss'),('gradient_penalty','discriminator/grad_penalty'),('weight_decay','discriminator/l2_loss'),('logit_decay','discriminator/l2_logit_loss')]}
                wrapped=getattr(agent,role) if role in {'actor','critic'} else getattr(component,'disc_critic' if role=='discriminator_critic' else role)
                agent._step_optimizer(loss,wrapped,optimizers[role],role)
                state=canonical(model.state_dict());gradients=canonical({name:value.grad for name,value in model.named_parameters()})
            else:
                if role=='actor':
                    loss,logs=trainer.actor_loss(batch);metrics={name:logs[name] for name in ['ppo','bounds','extra','entropy']}
                elif role=='critic':loss=trainer.value_loss(batch);metrics={}
                elif role=='discriminator_critic':loss=trainer.value_loss(batch,discriminator=True);metrics={}
                else:loss,metrics=trainer.discriminator_loss(batch)
                trainer._update(role,loss)
                state=model.state_dict();gradients={name:value.grad for name,value in model.named_parameters()}
            results[f'{iteration}/{role}']=tensors(dict(loss=loss,metrics=metrics,state=state,gradients=gradients,
                optimizer=optimizers[role].state_dict(),cuda_rng=torch.cuda.get_rng_state()))
    torch.save(results,args.output_dir/(args.worker+'.pt'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['upstream','artifact','output-dir']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--worker',choices=['upstream','native']);args=parser.parse_args()
    for name in ['upstream','artifact','output_dir']:setattr(args,name,getattr(args,name).resolve())
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];args.output_dir.mkdir(parents=True,exist_ok=True)
    path=args.output_dir/'source_parity.json'
    report=dict(complete=False,scope='Released G1 checkpoint and actual original PPO actor/L2C2, clipped task/discriminator value and AMP discriminator objectives, active gradients, running stats and actual Fabric optimizer steps versus native Trainer. Three synthetic B=4 minibatches including ordinary nonzero discriminator-value gradients and the original excessive-gradient guard; not rollout collection, replay sampling, full PPO epochs, simulator or full training protocol.',
        upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),
        artifact=json.loads((args.artifact/'provenance.json').read_text()),tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        env={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8','MUJOCO_GL':'egl'}
        subprocess.run([sys.executable,__file__,*sys.argv[1:],'--worker',kind],
            cwd=args.upstream if kind=='upstream' else root,env=env,check=True)
    import torch
    torch.set_num_threads(4)
    try:compare_full(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,torch_version=torch.__version__);path.write_text(json.dumps(report,indent=2)+'\n')
    print('Original G1 PPO/L2C2/AMP objectives, gradients and optimizer updates exact')


if __name__=='__main__':main()
