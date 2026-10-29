"""Original autoregressive generation and per-token diffusion, outside neural modules."""

import torch
from functools import lru_cache, partial
from motius.diffusion.motionstreamer import create_diffusion


@lru_cache(maxsize=4)
def _schedule(steps):
    return create_diffusion(timestep_respacing=str(steps), noise_schedule="cosine")


def predict_with_cfg(self, x, t, c, cfg_scale):
    half = x[: len(x) // 2]
    combined = torch.cat([half, half], dim=0)
    model_out = self.forward(combined, t, c)
    eps, rest = (model_out[:, : self.in_channels], model_out[:, self.in_channels :])
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    eps = torch.cat([half_eps, half_eps], dim=0)
    return torch.cat([eps, rest], dim=1)


def sample_token(head, z, temperature=1.0, cfg=1.0):
    if cfg != 1.0:
        noise = torch.randn(z.shape[0] // 2, head.in_channels).to(z.device)
        noise = torch.cat([noise, noise], dim=0)
        parameters = dict(c=z, cfg_scale=cfg)
        predict = partial(predict_with_cfg, head.net)
    else:
        noise = torch.randn(z.shape[0], head.in_channels).to(z.device)
        parameters = dict(c=z)
        predict = head.net.forward
    return _schedule(head.sampling_steps).p_sample_loop(
        predict,
        noise.shape,
        noise,
        clip_denoised=False,
        model_kwargs=parameters,
        progress=False,
        temperature=temperature,
    )


def sample_for_eval_CFG(
    self,
    text,
    length=196,
    tokenize_model=None,
    device=torch.device("cuda"),
    unit_length=4,
    cfg=4.0,
):
    max_token_len = length // unit_length
    for k in range(max_token_len):
        if k == 0:
            x = []
        else:
            x = xs
        feat_text = torch.from_numpy(tokenize_model.encode(text)).float()
        feat_text = feat_text.to(device)
        conditions = self.forward(x, feat_text)
        conditions = conditions[:, -1, :]
        empty_text = ""
        empty_feat_text = torch.from_numpy(tokenize_model.encode(empty_text)).float()
        empty_feat_text = empty_feat_text.unsqueeze(0)
        empty_feat_text = empty_feat_text.to(device)
        empty_conditions = self.forward(x, empty_feat_text)
        empty_conditions = empty_conditions[:, -1, :]
        temperature = 1.0
        if cfg != 1:
            mix_conditions = torch.cat([conditions, empty_conditions], dim=0)
            sampled_token_latent = sample_token(
                self.diff_loss, mix_conditions, temperature=temperature, cfg=cfg
            )
            scaled_logits, _ = sampled_token_latent.chunk(2, dim=0)
        else:
            scaled_logits = sample_token(
                self.diff_loss, conditions, temperature=temperature, cfg=1
            )
        scaled_logits = scaled_logits.unsqueeze(0)
        if k == 0:
            xs = scaled_logits
        else:
            xs = torch.cat((xs, scaled_logits), dim=1)
    return xs


def sample_for_eval_CFG_babel_inference_new(
    self,
    B_text,
    A_motion,
    if_categorial=False,
    length=78,
    clip_model=None,
    device=torch.device("cuda"),
    tokenizer="clip",
    unit_length=4,
    reference_end_token=None,
    cfg=4.5,
    threshold=3,
):
    B_token_length = length // unit_length
    if tokenizer == "clip":
        import clip

        A_text = clip.tokenize(A_text, truncate=True).to(device)
        A_feat_clip_text = clip_model.encode_text(A_text).float()
        B_text = clip.tokenize(B_text, truncate=True).to(device)
        B_feat_clip_text = clip_model.encode_text(B_text).float()
    elif tokenizer == "t5-xxl":
        B_feat_clip_text = torch.from_numpy(clip_model.encode(B_text)).float()
        B_feat_clip_text = B_feat_clip_text.to(device)
    empty_clip_text = ""
    if tokenizer == "clip":
        empty_text = clip.tokenize(empty_clip_text, truncate=True).to(device)
        empty_feat_clip_text = clip_model.encode_text(empty_text).float()
    elif tokenizer == "t5-xxl":
        empty_feat_clip_text = torch.from_numpy(
            clip_model.encode(empty_clip_text)
        ).float()
        empty_feat_clip_text = empty_feat_clip_text.unsqueeze(0)
        empty_feat_clip_text = empty_feat_clip_text.to(device)
    B_text_embeddings = self.transformer.cond_embed(B_feat_clip_text).unsqueeze(0)
    A_motion = A_motion.unsqueeze(0)
    A_motion_embeddings = self.transformer.wte(A_motion)
    B_motion = torch.tensor([]).to(device)
    attention_weights = []
    for k in range(B_token_length):
        if k == 0:
            x = torch.cat([B_text_embeddings, A_motion_embeddings], dim=1)
        else:
            x = xs
        conditions = self.forward_babel_eval(x, return_attention=False)
        conditions = conditions[:, -1, :]
        empty_feat_clip_text_embedding = self.transformer.cond_embed(
            empty_feat_clip_text
        ).unsqueeze(0)
        if k == 0:
            empty_input = torch.cat(
                [empty_feat_clip_text_embedding, A_motion_embeddings], dim=1
            )
            empty_conditions = self.forward_babel_eval(empty_input)
        else:
            B_motion_embeddings = self.transformer.wte(B_motion)
            empty_input = torch.cat(
                [
                    empty_feat_clip_text_embedding,
                    A_motion_embeddings,
                    B_motion_embeddings,
                ],
                dim=1,
            )
            empty_conditions = self.forward_babel_eval(empty_input)
        empty_conditions = empty_conditions[:, -1, :]
        temperature = 1.0
        mix_conditions = torch.cat([conditions, empty_conditions], dim=0)
        sampled_token_latent = sample_token(
            self.diff_loss, mix_conditions, temperature=temperature, cfg=cfg
        )
        if cfg != 1:
            scaled_logits, _ = sampled_token_latent.chunk(2, dim=0)
        else:
            scaled_logits = sampled_token_latent
        scaled_logits = scaled_logits.unsqueeze(0)
        if reference_end_token is not None:
            distance_l2 = torch.sqrt(
                torch.sum((scaled_logits - reference_end_token) ** 2)
            )
            print(distance_l2)
            if distance_l2 < threshold:
                break
        B_motion = torch.cat((B_motion, scaled_logits), dim=1)
        scaled_logits_embedding = self.transformer.wte(scaled_logits)
        xs = torch.cat((x, scaled_logits_embedding), dim=1)
    return (xs, B_motion)


def sample_for_eval_CFG_babel_inference_new_demo(
    self,
    B_text,
    A_motion,
    if_categorial=False,
    length=312,
    clip_model=None,
    device=torch.device("cuda"),
    tokenizer="clip",
    unit_length=4,
    reference_end_token=None,
    cfg=4.5,
    threshold=3,
    temperature=1.0,
):
    B_token_length = length // unit_length - A_motion.shape[0]
    if tokenizer == "clip":
        import clip

        A_text = clip.tokenize(A_text, truncate=True).to(device)
        A_feat_clip_text = clip_model.encode_text(A_text).float()
        B_text = clip.tokenize(B_text, truncate=True).to(device)
        B_feat_clip_text = clip_model.encode_text(B_text).float()
    elif tokenizer == "t5-xxl":
        B_feat_clip_text = torch.from_numpy(clip_model.encode(B_text)).float()
        B_feat_clip_text = B_feat_clip_text.to(device)
    empty_clip_text = ""
    if tokenizer == "clip":
        empty_text = clip.tokenize(empty_clip_text, truncate=True).to(device)
        empty_feat_clip_text = clip_model.encode_text(empty_text).float()
    elif tokenizer == "t5-xxl":
        empty_feat_clip_text = torch.from_numpy(
            clip_model.encode(empty_clip_text)
        ).float()
        empty_feat_clip_text = empty_feat_clip_text.unsqueeze(0)
        empty_feat_clip_text = empty_feat_clip_text.to(device)
    B_text_embeddings = self.transformer.cond_embed(B_feat_clip_text).unsqueeze(0)
    B_text_embeddings = B_text_embeddings.unsqueeze(0)
    A_motion = A_motion.unsqueeze(0)
    A_motion_embeddings = self.transformer.wte(A_motion)
    B_motion = torch.tensor([]).to(device)
    attention_weights = []
    for k in range(B_token_length):
        if k == 0:
            x = torch.cat([B_text_embeddings, A_motion_embeddings], dim=1)
        else:
            x = xs
        conditions = self.forward_babel_eval(x, return_attention=False)
        conditions = conditions[:, -1, :]
        empty_feat_clip_text_embedding = self.transformer.cond_embed(
            empty_feat_clip_text
        ).unsqueeze(0)
        if k == 0:
            empty_input = torch.cat(
                [empty_feat_clip_text_embedding, A_motion_embeddings], dim=1
            )
            empty_conditions = self.forward_babel_eval(empty_input)
        else:
            B_motion_embeddings = self.transformer.wte(B_motion)
            empty_input = torch.cat(
                [
                    empty_feat_clip_text_embedding,
                    A_motion_embeddings,
                    B_motion_embeddings,
                ],
                dim=1,
            )
            empty_conditions = self.forward_babel_eval(empty_input)
        empty_conditions = empty_conditions[:, -1, :]
        mix_conditions = torch.cat([conditions, empty_conditions], dim=0)
        sampled_token_latent = sample_token(
            self.diff_loss, mix_conditions, temperature=temperature, cfg=cfg
        )
        if cfg != 1:
            scaled_logits, _ = sampled_token_latent.chunk(2, dim=0)
        else:
            scaled_logits = sampled_token_latent
        scaled_logits = scaled_logits.unsqueeze(0)
        if reference_end_token is not None:
            distance_l2 = torch.sqrt(
                torch.sum((scaled_logits - reference_end_token) ** 2)
            )
            print(distance_l2)
            if distance_l2 < threshold and k > 10:
                break
        B_motion = torch.cat((B_motion, scaled_logits), dim=1)
        scaled_logits_embedding = self.transformer.wte(scaled_logits)
        xs = torch.cat((x, scaled_logits_embedding), dim=1)
    return (xs, B_motion)
