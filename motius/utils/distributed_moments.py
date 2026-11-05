"""Pairwise moment reduction for synchronized observation normalization."""

import torch


def combine_moments(means, variances, counts):
    counts = counts.float()
    while len(means) > 1:
        next_means = []
        next_variances = []
        next_counts = []
        for index in range(0, len(means), 2):
            if index + 1 < len(means):
                mean_a, var_a, count_a = means[index], variances[index], counts[index]
                mean_b, var_b, count_b = (
                    means[index + 1],
                    variances[index + 1],
                    counts[index + 1],
                )
                total = count_a + count_b
                delta = mean_b - mean_a
                mean = mean_a + delta * (count_b / total)
                moment = (
                    var_a * count_a
                    + var_b * count_b
                    + (delta**2) * (count_a * count_b / total)
                )
                next_means.append(mean)
                next_variances.append(moment / total)
                next_counts.append(total)
            else:
                next_means.append(means[index])
                next_variances.append(variances[index])
                next_counts.append(counts[index])
        means = next_means
        variances = next_variances
        counts = next_counts
    return means[0], torch.clamp(variances[0], min=0.0), counts[0].long()


def gather_moments(mean, variance, count):
    count = torch.as_tensor(count, device=mean.device, dtype=torch.long)
    size = torch.distributed.get_world_size()
    means = [torch.empty_like(mean) for _ in range(size)]
    variances = [torch.empty_like(variance) for _ in range(size)]
    counts = [torch.empty_like(count) for _ in range(size)]
    torch.distributed.all_gather(means, mean)
    torch.distributed.all_gather(variances, variance)
    torch.distributed.all_gather(counts, count)
    return combine_moments(means, variances, torch.stack(counts))
