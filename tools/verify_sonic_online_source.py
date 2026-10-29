"""Released SONIC online rollouts and PPO epochs against independent source code.

The controlled tensor environment exercises termination and timeout handling;
this certificate does not claim physical simulator equivalence.
"""

from pathlib import Path
from types import SimpleNamespace as NS
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify_sonic_network_source import SCHEMA, original_roots
from verify_sonic_training_source import canonical
from verify_prompthmr_video_parity import tensors, compare_full


class TensorEnvironment:
    """Deterministic nonzero inputs; all neural networks remain released models."""

    num_envs = 4

    def __init__(self):
        import torch

        self.state = torch.zeros(4, 29, device="cuda")
        self.tick = 0
        self.config = {}
        self.adaptive_syncs = []

    def sync_and_compute_adaptive_sampling(self, accelerator, sync_across_gpus=False):
        self.adaptive_syncs.append(bool(sync_across_gpus))

    def set_is_evaluating(self, **kwargs):
        pass

    def observations(self):
        import math
        import torch

        fields = []
        modes = torch.tensor(
            [[1, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 0]], device="cuda"
        ).float()
        for index, (name, shape) in enumerate(SCHEMA.items()):
            width = math.prod(shape)
            value = (
                torch.sin(
                    torch.arange(4 * width, device="cuda").reshape(4, width) * 0.017
                    + self.tick * 0.03
                    + index
                )
                * 0.12
                + self.state[:, :1] * 0.01
            )
            fields.append(modes if name == "encoder_index" else value)
        result = {"tokenizer_obs": torch.cat(fields, -1)}
        for name, width in [("actor_obs", 930), ("critic_obs", 1645)]:
            result[name] = (
                torch.cos(
                    torch.arange(4 * width, device="cuda").reshape(4, width) * 0.013
                    + self.tick * 0.02
                )
                * 0.12
                + self.state[:, :1] * 0.01
            )
        return result

    def reset_all(self):
        self.state.zero_()
        self.tick = 0
        return self.observations()

    def step(self, policy_state):
        import torch

        actions = policy_state["actions"]
        self.state = 0.93 * self.state + 0.02 * actions
        rewards = 0.2 - 0.001 * actions.square().sum(-1) + 0.01 * self.state[:, 0]
        self.tick += 1
        dones = torch.tensor(
            [self.tick % 4 == 0, self.tick % 5 == 0, False, self.tick % 7 == 0],
            device="cuda",
        )
        timeouts = dones & torch.tensor([False, True, False, False], device="cuda")
        self.state[dones] = 0
        return (
            self.observations(),
            rewards,
            dones,
            {"time_outs": timeouts, "episode": {}, "to_log": {}},
        )


def worker(args):
    import torch
    from accelerate import Accelerator
    from omegaconf import OmegaConf

    if os.environ.get("MOTIUS_REFERENCE_HOST"):

        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(731)
    accelerator = Accelerator(mixed_precision="no")
    checkpoint = args.artifact / "sonic_release/last.pt"
    environment = TensorEnvironment()
    if args.worker == "upstream":
        policy, critic = original_roots(
            args.upstream,
            checkpoint,
            checkpoint.parent / "config.yaml",
            include_auxiliary=True,
        )
        from gear_sonic.trl.trainer.ppo_trainer import PolicyAndValueWrapper
        from gear_sonic.trl.trainer.ppo_trainer_aux_loss import TRLAuxLossPPOTrainer

        model = PolicyAndValueWrapper(policy, critic)
        trainer = object.__new__(TRLAuxLossPPOTrainer)
        trainer.config = OmegaConf.create({})
        trainer.args = NS(
            cliprange_value=0.2,
            cliprange=0.2,
            vf_coef=1.0,
            learning_rate=2e-5,
            max_grad_norm=0.1,
            fp16=False,
        )
        trainer.accelerator = accelerator
        trainer.use_symmetry = False
        trainer.multi_critic_advantage_weights = None
        trainer.entropy_coef = 0.01
        trainer.desired_kl = 0.01
        trainer.adaptive_lr_min, trainer.adaptive_lr_max = 1e-5, 2e-4
        trainer.compute_imgaug_bc_loss = False
        trainer.compute_aux_loss, trainer.aux_loss_scale = True, 1.0
        trainer.model, trainer.use_apex = model, False
        trainer.policy_model, trainer.value_model = policy, critic
        trainer.env = environment
        trainer.num_steps_per_env, trainer.num_act, trainer.num_critics = 6, 29, 1
        trainer.algo_obs_dim_dict = {
            "actor_obs": 930,
            "critic_obs": 1645,
            "tokenizer_obs": 1750,
        }
        # The source model uses the observation group name "tokenizer".
        trainer.algo_obs_dim_dict["tokenizer"] = trainer.algo_obs_dim_dict.pop(
            "tokenizer_obs"
        )
        original_observations = environment.observations

        def source_observations():
            result = original_observations()
            result["tokenizer"] = result.pop("tokenizer_obs")
            return result

        environment.observations = source_observations
        trainer.learn_normalized_actions = False
        trainer.train_with_evaluating_env = False
        trainer.sync_advantage_normalization = True
        trainer.use_padding_mask = False
        trainer.gamma, trainer.lam = 0.99, 0.95
        trainer.state = NS(global_step=0)
        trainer.episode_env_tensors = NS(add=lambda values: None)
        trainer._setup_storage()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.0)
        trainer.optimizer = optimizer
        observations = environment.reset_all()
    else:
        from motius.models.sonic import SONICPolicyBundle
        from motius.simulators.rl.sonic_rollout import SonicRollout
        from motius.trainers.sonic.online import SONICOnlinePPOTrainer

        model = SONICPolicyBundle.from_original_checkpoint(checkpoint, device="cuda")
        policy, critic = model.policy, model.critic
        collector = SonicRollout(model, environment, SCHEMA, accelerator=accelerator)
        trainer = SONICOnlinePPOTrainer(
            model, collector=collector, learning_epochs=2, num_mini_batches=2
        )
        trainer.accelerator = accelerator
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.0)
        trainer.set_optimizers({"joint": optimizer})
    results = {}
    for iteration in range(2):
        torch.manual_seed(850 + iteration)
        if args.worker == "upstream":
            trainer.state.global_step = iteration
        else:
            trainer.runner = NS(global_step=iteration)
        if args.worker == "upstream":
            observations = trainer._rollout_step(model, observations)
            rollout = trainer._get_rollout_data(observations.keys())
            rollout["all_obs_dict"]["tokenizer_obs"] = rollout["all_obs_dict"].pop(
                "tokenizer"
            )
            names = {
                "all_obs_dict": "observations",
                "logprobs": "old_logprobs",
                "old_mu_batch": "old_mean",
                "old_sigma_batch": "old_sigma",
                "padding_mask_p1": "padding_mask_value",
            }
            canonical_rollout = {names.get(k, k): v for k, v in rollout.items()}
            # Snapshot before training updates or the next collection.
            results[str(iteration) + "/rollout"] = tensors(canonical_rollout)
            rollout["all_obs_dict"]["tokenizer"] = rollout["all_obs_dict"].pop(
                "tokenizer_obs"
            )
            trainer._train_mode()
            updates = []
            for epoch in range(2):
                order = torch.randperm(4, device="cuda")
                for start in range(0, 4, 2):
                    indices = order[start : start + 2]
                    batch = trainer._get_mb_rollout_data(rollout, indices)
                    with accelerator.accumulate(model):
                        forward = trainer._forward_model(model, batch)
                        output = trainer._compute_loss(forward, batch)
                        accelerator.backward(output["loss"])
                        norm = trainer._gradient_clipping()
                        optimizer.step()
                        optimizer.zero_grad()
                    updates.append(
                        tensors(
                            dict(
                                loss=output["loss"],
                                learning_rate=trainer.args.learning_rate,
                                indices=indices,
                                gradient_norm=norm,
                                auxiliary=output["aux_loss_dict"]["aux_losses_dict"],
                                **output["ppo_loss_dict"],
                            )
                        )
                    )
            trainer.sync_adaptive_sampling()
            state = canonical(policy, critic)
        else:
            rollout = collector.collect(6)
            results[str(iteration) + "/rollout"] = tensors(rollout)
            trainer.train_step(rollout)
            updates = tensors(trainer.last_updates)
            state = model.state_dict()
        results[str(iteration) + "/training"] = tensors(
            dict(
                updates=updates,
                state=state,
                optimizer=optimizer.state_dict(),
                cuda_rng=torch.cuda.get_rng_state(),
                environment_state=environment.state,
                adaptive_sync_flags=environment.adaptive_syncs,
            )
        )
        print(args.worker, "complete rollout and PPO iteration", iteration, flush=True)
    environment.adaptive_syncs.clear()
    for step in [0, 198, 199, 200, 399]:
        if args.worker == "upstream":
            trainer.state.global_step = step
            trainer.sync_adaptive_sampling()
        else:
            trainer.sync_sampling(step)
    results["sampling_boundary_flags"] = tensors(environment.adaptive_syncs)
    torch.save(results, args.output_dir / (args.worker + ".pt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["upstream", "artifact", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=["upstream", "native"])
    parser.add_argument("--reuse-results", action="store_true")
    args = parser.parse_args()
    for name in ["upstream", "artifact", "output_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    import torch

    report = {
        "complete": False,
        "tests": [],
        "scope": "Actual released SONIC policy/value and original TRL rollout, timeout "
        "bootstrap, GAE, environment-axis shuffling, PPO/auxiliary epochs, "
        "adaptive LR, global clipping, AdamW and running-stat updates. "
        "Two iterations, four controlled tensor environments, six steps, "
        "two epochs and two minibatches; single GPU FP32. Physical "
        "simulation, nondefault padding/symmetry and distributed updates excluded.",
    }
    path = args.output_dir / "source_parity.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for kind in [] if args.reuse_results else ["upstream", "native"]:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--upstream",
                str(args.upstream),
                "--artifact",
                str(args.artifact),
                "--output-dir",
                str(args.output_dir),
                "--worker",
                kind,
            ],
            cwd=ROOT,
            check=True,
        )
    try:
        compare_full(
            torch.load(args.output_dir / "upstream.pt", weights_only=True),
            torch.load(args.output_dir / "native.pt", weights_only=True),
            "",
            report["tests"],
        )
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
    report["complete"] = True
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("SONIC original online rollouts and PPO epochs exact")


if __name__ == "__main__":
    main()
