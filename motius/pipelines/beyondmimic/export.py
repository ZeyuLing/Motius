"""Deploy the neural actor and independent reference data to the source ONNX contract."""

from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
import numpy as np


class BeyondMimicDeploymentGraph(nn.Module):
    def __init__(self, bundle, reference_motion):
        super().__init__()
        network = getattr(bundle.network, 'module', bundle.network)
        self.actor = deepcopy(network.actor).cpu().eval()
        self.normalizer = deepcopy(bundle.actor_normalizer).cpu().eval()
        self.names = [
            'joint_pos',
            'joint_vel',
            'body_pos_w',
            'body_quat_w',
            'body_lin_vel_w',
            'body_ang_vel_w',
        ]
        for name in self.names:
            value = torch.as_tensor(reference_motion[name], dtype=torch.float32).cpu()
            self.register_buffer(name, value)
        self.time_step_total = len(self.joint_pos)
        if self.time_step_total == 0:
            raise ValueError('The deployment reference has no frames')

    def forward(self, observations, time_step):
        index = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self.actor(self.normalizer(observations)),
            self.joint_pos[index],
            self.joint_vel[index],
            self.body_pos_w[index],
            self.body_quat_w[index],
            self.body_lin_vel_w[index],
            self.body_ang_vel_w[index],
        )


def export_onnx_policy(bundle, reference_motion, output_file, *, metadata=None):
    import onnx

    graph = BeyondMimicDeploymentGraph(bundle, reference_motion).eval()
    file = Path(output_file)
    file.parent.mkdir(parents=True, exist_ok=True)
    observations = torch.zeros(1, graph.actor[0].in_features)
    time_step = torch.zeros(1, 1)
    torch.onnx.export(
        graph,
        (observations, time_step),
        str(file),
        opset_version=11,
        dynamo=False,
        input_names=['obs', 'time_step'],
        output_names=['actions', *graph.names],
        dynamic_axes={},
    )
    model = onnx.load(str(file))
    for name, value in (metadata or {}).items():
        field = model.metadata_props.add()
        field.key = name
        if isinstance(value, (tuple, list, np.ndarray)):
            field.value = ','.join(
                (
                    f'{item:.3f}'
                    if isinstance(item, (float, int, np.number))
                    else str(item)
                )
                for item in value
            )
        else:
            field.value = str(value)
    onnx.save(model, str(file))
    return file
