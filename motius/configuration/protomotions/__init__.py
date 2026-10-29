"""Read neural architecture metadata from an original ProtoMotions artifact."""

import ast
from pathlib import Path


def _typed(value):
    if isinstance(value, dict):
        return {key: _typed(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_typed(item) for item in value]
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    return value


def _configuration(value):
    import yaml

    if not isinstance(value, dict):
        value = yaml.safe_load(Path(value).read_text())
    return _typed(value)


def original_neural_options(weights, configuration):
    source = _configuration(configuration)['agent']['model']

    def mlp(config, prefix):
        if config['_target_'] != 'protomotions.agents.common.mlp.MLPWithConcat':
            raise ValueError(
                'Unsupported ProtoMotions neural module: ' + config['_target_']
            )
        if config.get('module_operations', [{}]) != [{}]:
            raise ValueError(
                'This checkpoint requires additional neural module operations'
            )
        if config.get('output_activation') is not None:
            raise ValueError('This checkpoint requires an output activation')
        layers = config['layers']
        activation = layers[0]['activation']
        if any(layer['activation'] != activation for layer in layers):
            raise ValueError('Mixed hidden activations are not supported')
        layer_norm = bool(layers[0]['use_layer_norm'])
        if any(layer['use_layer_norm'] for layer in layers[1:]):
            raise ValueError('Unexpected hidden layer normalization')
        return dict(
            input_dimension=weights[prefix + '.mlp.0.weight'].shape[1],
            hidden_dimensions=[layer['units'] for layer in layers],
            output_dimension=config['num_out'],
            input_keys=config['in_keys'],
            activation=activation,
            normalize=config['normalize_obs'],
            clamp=config['norm_clamp_value'],
            ema_decay=config.get('norm_ema_decay', 0.999),
            layer_norm=layer_norm,
        )

    actor = source['actor']
    options = dict(
        actor=dict(
            mean_options=mlp(actor['mu_model'], '_actor.mu'),
            initial_log_std=actor['actor_logstd'],
            learnable_std=actor['learnable_std'],
        ),
        critic=mlp(source['critic'], '_critic'),
    )
    for source_name, target_name in [
        ('discriminator', 'discriminator'),
        ('disc_critic', 'discriminator_critic'),
    ]:
        if source_name in source:
            models = source[source_name]['models']
            if len(models) != 1:
                raise ValueError(
                    'Expected a single published G1 ' + source_name + ' MLP'
                )
            options[target_name] = mlp(models[0], '_' + source_name + '.models.0')
    return options


def original_deployment_options(configuration):
    source = _configuration(configuration)
    keys = source['agent']['model']['actor']['mu_model']['in_keys']
    observations = {key: source['env']['observation_components'][key] for key in keys}
    allowed = {
        'compute_humanoid_reduced_coords_observations',
        'build_reduced_coords_target_poses',
        'compute_historical_actions_from_state',
    }
    for config in observations.values():
        if config['compute_func'] not in allowed:
            raise ValueError(
                'Unsupported actor observation function: ' + config['compute_func']
            )
    action = source['env']['action_config']
    if action['fn'] != 'bm_pd_action':
        raise ValueError('Unsupported source PD action: ' + action['fn'])
    simulator = source['simulator']['sim']
    return dict(
        observations=observations,
        action={key: value for key, value in action.items() if key != 'fn'},
        control_hz=simulator['fps'] / simulator['decimation'],
        future_steps=source['env']['control_components']['mimic']['future_steps'],
    )


def original_training_options(configuration):
    source = _configuration(configuration)['agent']
    if source['normalize_rewards']:
        raise ValueError(
            'Native G1 online collection currently requires the published unnormalized rewards'
        )
    if source['adaptive_lr']['enabled']:
        raise ValueError('The published G1 recipe disables adaptive learning rates')
    amp = source['amp_parameters']
    optimizers = {}
    for role, key in [
        ('actor', 'actor'),
        ('critic', 'critic'),
        ('discriminator', 'discriminator'),
        ('discriminator_critic', 'disc_critic'),
    ]:
        options = dict(source['model'][key + '_optimizer'])
        if options.pop('_target_') != 'torch.optim.Adam':
            raise ValueError('Native G1 recipe expects original Adam optimizers')
        optimizers[role] = options
    return dict(
        num_steps=source['num_steps'],
        batch_size=source['batch_size'],
        mini_epochs=source['num_mini_epochs'],
        max_steps=source['training_max_steps'],
        save_interval=source['save_last_checkpoint_every'],
        optimizers=optimizers,
        rollout=dict(
            gamma=source['gamma'],
            tau=source['tau'],
            task_weight=source['task_reward_w'],
            style_weight=amp['discriminator_reward_w'],
            advantage_normalization=source['advantage_normalization'],
            reward_threshold=amp['discriminator_reward_threshold'],
            max_bad_transitions=amp['discriminator_max_cumulative_bad_transitions'],
        ),
        replay=dict(
            capacity=amp['discriminator_replay_size'],
            keep_probability=amp['discriminator_replay_keep_prob'],
        ),
        trainer=dict(
            clip=source['e_clip'],
            entropy_coef=source['entropy_coef'],
            bounds_loss_coef=source['bounds_loss_coef'],
            clip_value_loss=source['clip_critic_loss'],
            l2c2_weight=(
                source['l2c2']['lambda_l2c2'] if source['l2c2']['enabled'] else None
            ),
            l2c2_pairs=source['l2c2']['obs_pairs'],
            max_grad_norm=source['gradient_clip_val'],
            actor_clip_frac_threshold=source['actor_clip_frac_threshold'],
            discriminator_batch_size=amp['discriminator_batch_size'],
            discriminator_grad_penalty=amp['discriminator_grad_penalty'],
            discriminator_weight_decay=amp['discriminator_weight_decay'],
            discriminator_logit_weight_decay=amp['discriminator_logit_weight_decay'],
            discriminator_optimization_ratio=amp['discriminator_optimization_ratio'],
            check_grad_mag=source['check_grad_mag'],
            fail_on_bad_grads=source['fail_on_bad_grads'],
        ),
    )
