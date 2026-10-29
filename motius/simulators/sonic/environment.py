"""Adapt the released G1 Isaac Lab environment to the native online collector."""

from collections import OrderedDict
import torch


class SonicG1Environment:
    def __init__(self, environment):
        self.env = environment
        environment.wrapper = self
        self.config = environment.cfg.config
        self.device, self.num_envs = environment.device, environment.num_envs
        self.motion_command = environment.command_manager.get_term("motion")
        self._motion_lib = self.motion_command.motion_lib
        self.is_evaluating = False
        self.schema = None

    def process_observations(self, observations):
        tokenizer = observations["tokenizer"]
        if not isinstance(tokenizer, dict):
            raise ValueError(
                "SONIC tokenizer observations must retain their named terms"
            )
        names = self.env.observation_manager._group_obs_term_names["tokenizer"]
        if self.schema is None:
            self.schema = OrderedDict(
                (name, tuple(tokenizer[name].shape[1:])) for name in names
            )
        return dict(
            actor_obs=observations["policy"],
            critic_obs=observations["critic"],
            tokenizer_obs=torch.cat(
                [tokenizer[name].reshape(self.num_envs, -1) for name in names], -1
            ),
        )

    def reset_all(self):
        observations, _ = self.env.reset()
        self.env.success_lift = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        return self.process_observations(observations)

    def step(self, policy_state):
        actions = policy_state["actions"]
        limit = self.config.get("action_clip_value")
        if limit is not None and limit > 0:
            actions = torch.clip(actions, -limit, limit)
        observations, rewards, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(torch.long)
        self.env.success_lift = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        extras["time_outs"], extras["episode"] = truncated, {}
        extras["to_log"] = {
            name: (
                value
                if isinstance(value, torch.Tensor)
                else torch.tensor(value, dtype=torch.float)
            )
            for name, value in extras["log"].items()
        }
        extras["env_actions"] = actions.detach().cpu()
        return self.process_observations(observations), rewards, dones, extras

    def set_is_evaluating(self, is_evaluating=False, global_rank=0, **kwargs):
        self.is_evaluating = is_evaluating
        self.motion_command.set_is_evaluating(is_evaluating)
        if is_evaluating:
            self._motion_lib.load_motions_for_evaluation(
                start_idx=global_rank * self.num_envs
            )
            self.reset_all()

    def resample_motion(self):
        changed = self._motion_lib.load_motions_for_training(
            max_num_seqs=min(self.num_envs, self.motion_command.max_num_load_motions)
        )
        if changed:
            self.reset_all()

    def sync_and_compute_adaptive_sampling(self, accelerator, sync_across_gpus=False):
        self._motion_lib.sync_and_compute_adaptive_sampling(
            accelerator, sync_across_gpus=sync_across_gpus
        )

    def get_env_state_dict(self):
        return {"motion_lib": self._motion_lib.get_state_dict()}

    def load_env_state_dict(self, state):
        if "motion_lib" in state:
            self._motion_lib.load_state_dict(state["motion_lib"])
            if self._motion_lib.use_adaptive_sampling:
                self.resample_motion()

    def close(self):
        self.env.close()


def build_sonic_environment(configuration, **kwargs):
    from isaaclab.envs import ManagerBasedRLEnv
    from motius.configuration.sonic.environment import load_manager_configuration

    config = load_manager_configuration(configuration, **kwargs)
    return SonicG1Environment(ManagerBasedRLEnv(config))
