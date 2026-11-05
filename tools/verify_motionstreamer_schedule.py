"""Compare the native optimizer/LR path with the original scheduler and resume it."""

from pathlib import Path
import argparse
import ast
import copy
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
    from motius.runner.accelerate_runner import AccelerateRunner

    file = args.upstream / "train_motionstreamer.py"
    node = next(
        item
        for item in ast.parse(file.read_text()).body
        if isinstance(item, ast.ClassDef) and item.name == "WarmupCosineDecayScheduler"
    )
    context = dict(LambdaLR=LambdaLR, CosineAnnealingLR=CosineAnnealingLR)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(file), "exec"), context)
    parameters = [torch.nn.Parameter(torch.tensor([0.17, -0.21])) for _ in range(2)]
    optimizers = [
        torch.optim.AdamW([parameter], lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-6)
        for parameter in parameters
    ]
    original = context["WarmupCosineDecayScheduler"](optimizers[0], 5, 23)
    config = dict(
        type="motius.trainers.motionstreamer.schedule.MotionStreamerWarmupCosine",
        warmup_iters=5,
        total_iters=23,
    )
    native = AccelerateRunner._build_single_scheduler(config, optimizers[1], 24)
    rates = []
    for index in range(24):
        used = [optimizer.param_groups[0]["lr"] for optimizer in optimizers]
        for parameter, optimizer in zip(parameters, optimizers):
            parameter.grad = torch.tensor([0.01 + 0.001 * index, -0.017])
            optimizer.step()
        original.step(index)
        native.step()
        assert used[0] == used[1]
        assert (
            optimizers[0].param_groups[0]["lr"] == optimizers[1].param_groups[0]["lr"]
        )
        assert torch.equal(parameters[0], parameters[1])
        rates.append(
            dict(
                iteration=index, used=used[0], next=optimizers[1].param_groups[0]["lr"]
            )
        )
        if index == 8:
            optimizer_state = copy.deepcopy(optimizers[1].state_dict())
            scheduler_state = copy.deepcopy(native.state_dict())
            restored = torch.nn.Parameter(parameters[1].detach().clone())
            optimizer = torch.optim.AdamW(
                [restored], lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-6
            )
            native = AccelerateRunner._build_single_scheduler(config, optimizer, 24)
            optimizer.load_state_dict(optimizer_state)
            native.load_state_dict(scheduler_state)
            parameters[1], optimizers[1] = restored, optimizer
    # TAE uses a different, exclusive warmup loop before its milestone clock.
    file = args.upstream / "train_causal_TAE.py"
    node = next(
        item
        for item in ast.parse(file.read_text()).body
        if isinstance(item, ast.FunctionDef) and item.name == "update_lr_warm_up"
    )
    context = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(file), "exec"), context)
    from torch.optim.lr_scheduler import MultiStepLR

    parameters = [torch.nn.Parameter(torch.tensor([0.17, -0.21])) for _ in range(2)]
    optimizers = [
        torch.optim.AdamW([parameter], lr=5e-5, betas=(0.9, 0.99), weight_decay=0.0)
        for parameter in parameters
    ]
    original = MultiStepLR(optimizers[0], milestones=[3, 7], gamma=0.05)
    config = dict(
        type="motius.trainers.motionstreamer.schedule.MotionStreamerTAESchedule",
        warmup_iters=5,
        milestones=[3, 7],
        gamma=0.05,
    )
    native = AccelerateRunner._build_single_scheduler(config, optimizers[1], 14)
    tae_rates = []
    for index in range(14):
        if index < 4:
            context["update_lr_warm_up"](optimizers[0], index + 1, 5, 5e-5)
        used = [optimizer.param_groups[0]["lr"] for optimizer in optimizers]
        for parameter, optimizer in zip(parameters, optimizers):
            parameter.grad = torch.tensor([0.01 + 0.001 * index, -0.017])
            optimizer.step()
        if index >= 4:
            original.step()
        native.step()
        assert used[0] == used[1], (index, used)
        assert torch.equal(parameters[0], parameters[1]), index
        tae_rates.append(dict(iteration=index, used=used[0]))
        if index in [1, 6]:
            optimizer_state = copy.deepcopy(optimizers[1].state_dict())
            scheduler_state = copy.deepcopy(native.state_dict())
            restored = torch.nn.Parameter(parameters[1].detach().clone())
            optimizer = torch.optim.AdamW(
                [restored], lr=5e-5, betas=(0.9, 0.99), weight_decay=0.0
            )
            native = AccelerateRunner._build_single_scheduler(config, optimizer, 14)
            optimizer.load_state_dict(optimizer_state)
            native.load_state_dict(scheduler_state)
            parameters[1], optimizers[1] = restored, optimizer
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                complete=True,
                scope="Actual original AR warmup/cosine and TAE exclusive warmup/milestone schedulers with AdamW versus native scheduler built through the shared Runner, including the original inclusive final iteration. Native optimizer/scheduler restored after update nine and remains identical to uninterrupted original trajectory. Tiny real parameter optimizer probe; not a full model training certificate.",
                exact_parameter_and_learning_rate_trajectory=True,
                tests=rates,
                tae_tests=tae_rates,
            ),
            indent=2,
        )
        + "\n"
    )
    print("Original/native optimizer/LR schedule and resumed trajectory exact")


if __name__ == "__main__":
    main()
