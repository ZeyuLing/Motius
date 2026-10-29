"""Build the inference scheduler outside the neural ModelBundle."""


def build_scheduler(sched_cfg: dict):
    from diffusers import LCMScheduler

    return LCMScheduler(
        num_train_timesteps=sched_cfg["num_train_timesteps"],
        beta_start=sched_cfg["beta_start"],
        beta_end=sched_cfg["beta_end"],
        beta_schedule=sched_cfg["beta_schedule"],
        clip_sample=sched_cfg["clip_sample"],
        set_alpha_to_one=sched_cfg["set_alpha_to_one"],
    )
