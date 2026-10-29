"""Compare native BeyondMimic policy/PPO with its independent RSL-RL 2.3.1 dependency."""
from __future__ import annotations
import argparse,ast,json,os,subprocess,sys
from pathlib import Path
from verify_prompthmr_video_parity import tensors,compare_full


def published_options(upstream):
    path=upstream/'source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py'
    cls=next(node for node in ast.parse(path.read_text()).body if isinstance(node,ast.ClassDef) and node.name=='G1FlatPPORunnerCfg')
    calls={node.targets[0].id:node.value for node in cls.body if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and isinstance(node.value,ast.Call)}
    values=lambda call:{item.arg:ast.literal_eval(item.value) for item in call.keywords}
    return values(calls['policy']),values(calls['algorithm'])


def worker(args):
    import torch
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    neural,options=published_options(args.upstream)
    neural.update(num_actor_obs=160,num_critic_obs=286,num_actions=29)
    torch.manual_seed(271)
    if args.worker=='upstream':
        sys.path.insert(0,str(args.rsl_rl))
        from rsl_rl.modules import ActorCritic,EmpiricalNormalization
        from rsl_rl.algorithms import PPO
        policy=ActorCritic(**neural).cuda()
        actor_norm=EmpiricalNormalization([160],until=1e8).cuda();critic_norm=EmpiricalNormalization([286],until=1e8).cuda()
        torch.save(dict(model_state_dict=policy.state_dict(),obs_norm_state_dict=actor_norm.state_dict(),privileged_obs_norm_state_dict=critic_norm.state_dict()),args.output_dir/'initial_source_checkpoint.pt')
        trainer=PPO(policy,**options,device='cuda');trainer.init_storage('rl',4,24,[160],[286],[29])
    else:
        from motius.models.beyondmimic import BeyondMimicPolicyBundle
        from motius.trainers.beyondmimic import BeyondMimicTrainer
        from motius.simulators.rl import generalized_advantage_estimation
        bundle=BeyondMimicPolicyBundle(neural,checkpoint=args.output_dir/'initial_source_checkpoint.pt',device='cuda').train()
        policy=bundle.network;actor_norm=bundle.actor_normalizer;critic_norm=bundle.critic_normalizer
        trainer=BeyondMimicTrainer(bundle,**options);trainer.set_optimizers({'policy':torch.optim.Adam(bundle.trainable_parameters(),lr=options['learning_rate'])})
    output={}
    for iteration in range(2):
        torch.manual_seed(370+iteration)
        raw_obs=torch.randn(24,4,160,device='cuda');raw_critic=torch.randn(24,4,286,device='cuda')
        actions=torch.randn(24,4,29,device='cuda')*.2;rewards=torch.randn(24,4,1,device='cuda')*.1
        dones=torch.rand(24,4,1,device='cuda')>.85;last_values=torch.randn(4,1,device='cuda')*.1
        with torch.no_grad():
            observations=torch.stack([actor_norm(value) for value in raw_obs]);critic_observations=torch.stack([critic_norm(value) for value in raw_critic])
            if args.worker=='upstream':
                mean=policy.actor(observations);sigma=policy.std.expand_as(mean);values=policy.critic(critic_observations)
            else:
                prediction=bundle.predict_policy(observations,critic_observations);mean=prediction['mean'];sigma=prediction['std'];values=prediction['value']
            old_mu=mean+.01;old_sigma=sigma*.99
            old_log_prob=torch.distributions.Normal(old_mu,old_sigma,validate_args=False).log_prob(actions).sum(dim=-1,keepdim=True)
        rollout=dict(observations=observations,critic_observations=critic_observations,actions=actions,values=values,
            actions_log_prob=old_log_prob,mu=old_mu,sigma=old_sigma)
        if args.worker=='upstream':
            storage=trainer.storage
            for key,value in {**rollout,'rewards':rewards,'dones':dones}.items():
                target='privileged_observations' if key=='critic_observations' else key
                getattr(storage,target).copy_(value)
            storage.compute_returns(last_values,options['gamma'],options['lam'])
            returns=storage.returns.clone();advantages=storage.advantages.clone()
            torch.manual_seed(595+iteration);metrics=trainer.update()
            metrics={key:metrics[source] for key,source in [('value','value_function'),('surrogate','surrogate'),('entropy','entropy')]}
            optimizer=trainer.optimizer
        else:
            returns,advantages=generalized_advantage_estimation(rewards,dones,values,last_values,gamma=options['gamma'],lam=options['lam'])
            rollout.update(returns=returns,advantages=advantages)
            torch.manual_seed(595+iteration);values_out=trainer.train_step(rollout)
            metrics={key:values_out['ppo_'+key] for key in ['value','surrogate','entropy']};optimizer=trainer.get_optimizer('policy')
        output[str(iteration)]=tensors(dict(returns=returns,advantages=advantages,metrics=metrics,learning_rate=trainer.learning_rate,
            state=policy.state_dict(),normalizers=dict(actor=actor_norm.state_dict(),critic=critic_norm.state_dict()),
            gradients={key:value.grad for key,value in policy.named_parameters()},optimizer=optimizer.state_dict(),cuda_rng=torch.cuda.get_rng_state()))
    torch.save(output,args.output_dir/(args.worker+'.pt'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['upstream','rsl-rl','output-dir']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--worker',choices=['upstream','native']);args=parser.parse_args()
    for name in ['upstream','rsl_rl','output_dir']:setattr(args,name,getattr(args,name).resolve())
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];args.output_dir.mkdir(parents=True,exist_ok=True)
    path=args.output_dir/'source_parity.json';report=dict(complete=False,comparison='Original BeyondMimic G1 PPO configuration and independent RSL-RL 2.3.1 policy, normalization, GAE, all PPO epochs, adaptive learning rate and Adam states versus native Bundle/Trainer',scope='Two synthetic 24x4 rollouts with original seeded neural weights; not a released pretrained G1 policy, physical simulator or full motion-dataset reproduction',upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),rsl_rl_commit=subprocess.check_output(['git','-C',str(args.rsl_rl),'rev-parse','HEAD'],text=True).strip(),tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        env={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
        subprocess.run([sys.executable,__file__,*sys.argv[1:],'--worker',kind],cwd=args.upstream if kind=='upstream' else root,env=env,check=True)
    import torch
    try:compare_full(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,torch_version=torch.__version__);path.write_text(json.dumps(report,indent=2)+'\n')
    print('All original BeyondMimic/RSL-RL PPO comparisons exact')

if __name__=='__main__':main()
