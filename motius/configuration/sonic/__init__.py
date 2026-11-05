"""Extract only neural architecture settings from a public SONIC artifact."""

from pathlib import Path


def original_neural_options(states, configuration):
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(configuration))
    actor = config.algo.config.actor
    backbone = actor.backbone
    policy = states.get("policy_state_dict", states.get("actor_model_state_dict"))
    critic = states.get("value_state_dict", states.get("critic_model_state_dict"))
    if policy is None or critic is None:
        raise ValueError("SONIC requires both published policy and value neural roots")

    def mlp(weights, prefix, parameters):
        target = str(parameters["_target_"])
        if not target.endswith((".modules.BaseModule", ".base_module.BaseModule")):
            raise ValueError("Unsupported SONIC neural class: " + target)
        layer = parameters.module_config_dict.layer_config
        if layer.type != "MLP":
            raise ValueError("Unsupported SONIC backbone layer: " + str(layer.type))
        keys = sorted(
            [
                name
                for name in weights
                if name.startswith(prefix + ".module.") and name.endswith(".weight")
            ],
            key=lambda name: int(name.split(".")[-2]),
        )
        matrices = [weights[name] for name in keys]
        if len(matrices) < 2:
            raise ValueError("Missing SONIC MLP matrices: " + prefix)
        hidden = [matrix.shape[0] for matrix in matrices[:-1]]
        if hidden != list(layer.hidden_dims):
            raise ValueError(
                "SONIC architecture metadata does not match its checkpoint: " + prefix
            )
        return dict(
            input_dimension=matrices[0].shape[1],
            hidden_dimensions=hidden,
            output_dimension=matrices[-1].shape[0],
            input_frames=parameters.get("num_input_temporal_dims"),
            output_frames=parameters.get("num_output_temporal_dims"),
            activation=str(layer.activation),
        )

    encoders = {
        name: mlp(policy, "actor_module.encoders." + name, item.params)
        for name, item in backbone.encoders.items()
    }
    decoders = {
        name: mlp(policy, "actor_module.decoders." + name, item.params)
        for name, item in backbone.decoders.items()
    }
    if str(backbone.quantizer["_target_"]) != "vector_quantize_pytorch.FSQ":
        raise ValueError("Unsupported SONIC quantizer")
    levels = backbone.fsq_level_list
    levels = (
        [int(levels)] * int(backbone.num_fsq_levels)
        if isinstance(levels, int)
        else list(levels)
    )
    return dict(
        policy=dict(
            encoders=encoders,
            decoders=decoders,
            levels=levels,
            num_actions=policy["std"].numel(),
            initial_std=float(config.algo.config.init_noise_std),
        ),
        critic=dict(
            network_options=mlp(
                critic, "critic_module", config.algo.config.critic.backbone
            ),
            normalize=bool(config.algo.config.critic.running_mean_std),
        ),
    )


def original_training_options(configuration):
    """Read the numeric released TRL protocol; no original trainer is constructed."""
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(configuration))
    algo = config.algo.config
    trl = config.algo.trl
    backbone = algo.actor.backbone
    return dict(
        trainer=dict(
            clip_range=float(trl.get("cliprange", 0.2)),
            clip_range_value=float(trl.get("cliprange_value", 0.2)),
            value_coefficient=float(trl.get("vf_coef", 1.0)),
            entropy_coefficient=float(algo.entropy_coef),
            auxiliary_coefficients=OmegaConf.to_container(
                backbone.aux_loss_coef, resolve=True
            ),
            auxiliary_scale=float(algo.get("aux_loss_scale", 1.0)),
            learning_rate=float(trl.learning_rate),
            desired_kl=algo.desired_kl,
            adaptive_lr_min=float(algo.adaptive_lr_min),
            adaptive_lr_max=float(algo.adaptive_lr_max),
            learning_epochs=int(trl.num_ppo_epochs),
            num_mini_batches=int(trl.num_mini_batches),
            micro_batch_size=trl.get("per_device_train_batch_size"),
            max_grad_norm=float(trl.max_grad_norm),
            shuffle_every_epoch=bool(algo.ppo_shuffle_every_epoch),
            sync_running_mean_std_freq=int(
                config.manager_env.config.get("sync_running_mean_std_freq", 1)
            ),
            sync_adaptive_sampling_freq=int(
                config.manager_env.config.get(
                    "sync_adaptive_sampling_all_gpus_freq", 200
                )
            ),
        ),
        optimizer=dict(
            lr=float(trl.learning_rate),
            betas=(
                float(trl.get("adam_beta1", 0.9)),
                float(trl.get("adam_beta2", 0.999)),
            ),
            eps=float(trl.get("adam_epsilon", 1e-8)),
            weight_decay=float(trl.get("weight_decay", 0.0)),
        ),
        rollout=dict(
            gamma=float(trl.gamma),
            lam=float(trl.lam),
            sync_advantage_normalization=bool(algo.sync_advantage_normalization),
            use_padding_mask=bool(algo.use_padding_mask),
        ),
        steps=int(algo.num_steps_per_env),
        iterations=int(trl.num_total_batches),
        gradient_accumulation_steps=int(trl.get("gradient_accumulation_steps", 1)),
        save_interval=int(algo.save_interval),
        resample_frequency=int(config.callbacks.im_resample.motion_resample_frequency),
        skip_resample_frequency=config.callbacks.im_resample.get(
            "skip_resample_frequency"
        ),
    )
