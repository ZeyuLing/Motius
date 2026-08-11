# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MotionCLIP model configuration"""

# Compatibility imports for different transformers versions

from transformers import logging
from transformers.configuration_utils import PretrainedConfig


logger = logging.get_logger(__name__)


class MotionCLIPTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CLIPTextModel`]. It is used to instantiate a CLIP
    text encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the text encoder of the CLIP
    [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 49408):
            Vocabulary size of the CLIP text model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`CLIPModel`].
        hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and motion projection layers.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        max_position_embeddings (`int`, *optional*, defaults to 77):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 49406):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 49407):
            End of stream token id.

    Example:

    ```python
    >>> from transformers import CLIPTextConfig, CLIPTextModel

    >>> # Initializing a CLIPTextConfig with openai/clip-vit-base-patch32 style configuration
    >>> configuration = CLIPTextConfig()

    >>> # Initializing a CLIPTextModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
    >>> model = CLIPTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "clip_text_model"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=49408,
        hidden_size=512,
        intermediate_size=2048,
        projection_dim=512,
        num_hidden_layers=12,
        num_attention_heads=8,
        max_position_embeddings=256,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        # This differs from `CLIPTokenizer`'s default and from openai/clip
        # See https://github.com/huggingface/transformers/pull/24773#issuecomment-1632287538
        pad_token_id=1,
        bos_token_id=49406,
        eos_token_id=49407,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout


class MotionCLIPMotionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MotionCLIPMotionModel`]. It is used to instantiate a
    MotionCLIP motion encoder according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and motion projection layers.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        motion_dim (`int`, *optional*, defaults to 263):
            The input dimension of each motion frame (e.g., joint positions, rotations).
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length (number of frames) that this model might ever be used with.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from mmotion.models.transformers.motion_clip import MotionCLIPMotionConfig, MotionCLIPMotionModel

    >>> # Initializing a MotionCLIPMotionConfig
    >>> configuration = MotionCLIPMotionConfig()

    >>> # Initializing a MotionCLIPMotionModel (with random weights)
    >>> model = MotionCLIPMotionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "motion_clip_motion_model"
    base_config_key = "motion_config"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        projection_dim=512,
        num_hidden_layers=12,
        num_attention_heads=12,
        motion_dim=263,
        max_position_embeddings=512,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.motion_dim = motion_dim
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act


class MotionCLIPConfig(PretrainedConfig):
    r"""
    [`CLIPConfig`] is the configuration class to store the configuration of a [`CLIPModel`]. It is used to instantiate
    a CLIP model according to the specified arguments, defining the text model and motion model configs. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the CLIP
    [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`CLIPTextConfig`].
        motion_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`CLIPMotionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and motion projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The initial value of the *logit_scale* parameter. Default is used as per the original CLIP implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import CLIPConfig, CLIPModel

    >>> # Initializing a CLIPConfig with openai/clip-vit-base-patch32 style configuration
    >>> configuration = CLIPConfig()

    >>> # Initializing a CLIPModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
    >>> model = CLIPModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a CLIPConfig from a CLIPTextConfig and a CLIPMotionConfig
    >>> from transformers import CLIPTextConfig, CLIPMotionConfig

    >>> # Initializing a CLIPText and CLIPMotion configuration
    >>> config_text = CLIPTextConfig()
    >>> config_motion = CLIPMotionConfig()

    >>> config = CLIPConfig(text_config=config_text, motion_config=config_motion)
    ```"""

    model_type = "clip"
    sub_configs = {
        "text_config": MotionCLIPTextConfig,
        "motion_config": MotionCLIPMotionConfig,
    }

    def __init__(
        self,
        text_config=None,
        motion_config=None,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        **kwargs,
    ):
        # If `_config_dict` exist, we use them for the backward compatibility.
        # We pop out these 2 attributes before calling `super().__init__` to avoid them being saved (which causes a lot
        # of confusion!).
        text_config_dict = kwargs.pop("text_config_dict", None)
        motion_config_dict = kwargs.pop("motion_config_dict", None)

        # Instead of simply assigning `[text|motion]_config_dict` to `[text|motion]_config`, we use the values in
        # `[text|motion]_config_dict` to update the values in `[text|motion]_config`. The values should be same in most
        # cases, but we don't want to break anything regarding `_config_dict` that existed before commit `8827e1b2`.
        if text_config_dict is not None:
            if text_config is None:
                text_config = {}

            # This is the complete result when using `text_config_dict`.
            _text_config_dict = MotionCLIPTextConfig(**text_config_dict).to_dict()

            # Give a warning if the values exist in both `_text_config_dict` and `text_config` but being different.
            for key, value in _text_config_dict.items():
                if (
                    key in text_config
                    and value != text_config[key]
                    and key != "transformers_version"
                ):
                    # If specified in `text_config_dict`
                    if key in text_config_dict:
                        message = (
                            f"`{key}` is found in both `text_config_dict` and `text_config` but with different values. "
                            f'The value `text_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`text_config_dict` is provided which will be used to initialize `CLIPTextConfig`. The "
                            f'value `text_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `text_config` with the ones in `_text_config_dict`.
            text_config.update(_text_config_dict)

        if motion_config_dict is not None:
            if motion_config is None:
                motion_config = {}

            # This is the complete result when using `motion_config_dict`.
            _motion_config_dict = MotionCLIPMotionConfig(**motion_config_dict).to_dict()
            # convert keys to string instead of integer
            if "id2label" in _motion_config_dict:
                _motion_config_dict["id2label"] = {
                    str(key): value
                    for key, value in _motion_config_dict["id2label"].items()
                }

            # Give a warning if the values exist in both `_motion_config_dict` and `motion_config` but being different.
            for key, value in _motion_config_dict.items():
                if (
                    key in motion_config
                    and value != motion_config[key]
                    and key != "transformers_version"
                ):
                    # If specified in `motion_config_dict`
                    if key in motion_config_dict:
                        message = (
                            f"`{key}` is found in both `motion_config_dict` and `motion_config` but with different "
                            f'values. The value `motion_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`motion_config_dict` is provided which will be used to initialize `CLIPMotionConfig`. "
                            f'The value `motion_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `motion_config` with the ones in `_motion_config_dict`.
            motion_config.update(_motion_config_dict)

        if text_config is None:
            text_config = MotionCLIPTextConfig()
            logger.info(
                "`text_config` is `None`. initializing the `MotionCLIPTextConfig` with default values."
            )
        elif isinstance(text_config, dict):
            text_config = MotionCLIPTextConfig(**text_config)

        if motion_config is None:
            motion_config = MotionCLIPMotionConfig()
            logger.info(
                "`motion_config` is `None`. initializing the `MotionCLIPMotionConfig` with default values."
            )
        elif isinstance(motion_config, dict):
            motion_config = MotionCLIPMotionConfig(**motion_config)

        self.text_config = text_config
        self.motion_config = motion_config

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0
        super().__init__(**kwargs)


__all__ = ["MotionCLIPConfig", "MotionCLIPTextConfig", "MotionCLIPMotionConfig"]
