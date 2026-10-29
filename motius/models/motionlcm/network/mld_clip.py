"""Frozen tensor-only text towers shared by MotionLCM and MLD."""

from pathlib import Path
import torch
from torch import nn


class MldTextEncoder(nn.Module):
    def __init__(self, modelpath: str, last_hidden_state: bool = False):
        super().__init__()
        from transformers import (
            AutoConfig,
            AutoModel,
            CLIPTextModel,
            CLIPTextModelWithProjection,
        )

        self.modelpath = str(modelpath)
        config = AutoConfig.from_pretrained(modelpath)
        clip_text = config.model_type in {"clip", "clip_text_model"}
        sentence_t5 = not clip_text and (
            "t5" in self.modelpath.lower()
            or (Path(modelpath) / "modules.json").is_file()
        )
        if clip_text:
            cls = CLIPTextModel if last_hidden_state else CLIPTextModelWithProjection
            self.text_model = cls.from_pretrained(modelpath)
            self.name = "clip_hidden" if last_hidden_state else "clip"
            self.text_encoded_dim = self.text_model.config.hidden_size
        elif sentence_t5:
            from motius.configuration.motionstreamer.text import load_text_network

            self.text_model, source = load_text_network(modelpath, "cpu")
            self.modelpath = str(source)
            self.name = "t5"
            self.text_encoded_dim = self.text_model.projection.out_features
        else:
            self.text_model = AutoModel.from_pretrained(modelpath)
            if "clip" in self.modelpath.lower():
                self.name = "clip_hidden" if last_hidden_state else "clip"
                self.text_encoded_dim = self.text_model.config.text_config.hidden_size
            elif "bert" in self.modelpath.lower():
                self.name = "bert"
                self.text_encoded_dim = self.text_model.config.hidden_size
            else:
                raise ValueError(f"Model {modelpath} not supported")

    def forward(self, features):
        device = next(self.text_model.parameters()).device
        features = {name: value.to(device) for name, value in features.items()}
        if self.name == "clip":
            return self.text_model(
                input_ids=features["input_ids"]
            ).text_embeds.unsqueeze(1)
        if self.name == "clip_hidden":
            return self.text_model(input_ids=features["input_ids"]).last_hidden_state
        if self.name == "bert":
            return self.text_model(**features).last_hidden_state
        return self.text_model(
            input_ids=features["input_ids"], attention_mask=features["attention_mask"]
        ).unsqueeze(1)

    def save_pretrained(self, directory, **kwargs):
        from motius.configuration.motionlcm_text import save_text_encoder

        return save_text_encoder(self, directory)
