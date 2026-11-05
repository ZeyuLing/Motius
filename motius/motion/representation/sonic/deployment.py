"""Observation packing for the published SONIC deployment graph.

The released ONNX graph reads its mode from column zero and slices the
remaining flat observations before reshaping. In particular its SMPL branch
uses 72 + 6 + 6 features per frame; the training schema uses 69 + 6 + 9.
Keep that distinction explicit instead of inferring layout from network sizes.
"""

import torch


def released_deployment_features(observations):
    if observations.ndim != 2 or observations.shape[-1] != 1762:
        raise ValueError(
            'Released SONIC encoder observations must have shape [B, 1762]'
        )
    batch = observations.shape[0]
    mode = observations[:, 0].long()
    if ((mode < 0) | (mode > 2)).any():
        raise ValueError('Released SONIC mode must be 0 (G1), 1 (teleop), or 2 (SMPL)')
    features = {
        'g1': torch.cat(
            [
                observations[:, 4:584].reshape(batch, 1, 10, 58),
                observations[:, 601:661].reshape(batch, 1, 10, 6),
            ],
            -1,
        ),
        'teleop': torch.cat(
            [
                observations[:, 661:901],
                observations[:, 901:910],
                observations[:, 910:922],
                observations[:, 595:601],
            ],
            -1,
        )[:, None],
        'smpl': torch.cat(
            [
                observations[:, 922:1642].reshape(batch, 1, 10, 72),
                observations[:, 1642:1702].reshape(batch, 1, 10, 6),
                observations[:, 1702:1762].reshape(batch, 1, 10, 6),
            ],
            -1,
        ),
    }
    masks = {name: mode == index for index, name in enumerate(features)}
    return features, masks
