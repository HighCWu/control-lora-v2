from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from diffusers.configuration_utils import register_to_config
from diffusers.utils import logging
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.models.controlnet import ControlNetModel
from diffusers.models.lora import (
    LoRACompatibleConv, 
    LoRACompatibleLinear, 
    LoRAConv2dLayer, 
    LoRALinearLayer,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _tie_weights(source_module: nn.Module, target_module: nn.Module):
    weight_names = [name for name, _ in source_module.named_parameters()]
    for weight_name in weight_names:
        branches = weight_name.split('.')
        base_weight_name = branches.pop(-1)
        source_parent_module = source_module
        target_parent_module = target_module
        for branch in branches:
            source_parent_module = getattr(source_parent_module, branch)
            target_parent_module = getattr(target_parent_module, branch)
        weight = getattr(source_parent_module, base_weight_name)
        setattr(target_parent_module, base_weight_name, weight)


class ControlLoRAModel(ControlNetModel):
    """
    A ControlLoRA model.

    Args:
        in_channels (`int`, defaults to 4):
            The number of channels in the input sample.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, defaults to 0):
            The frequency shift to apply to the time embedding.
        down_block_types (`tuple[str]`, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use.
        only_cross_attention (`Union[bool, Tuple[bool]]`, defaults to `False`):
        block_out_channels (`tuple[int]`, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, defaults to 2):
            The number of layers per block.
        downsample_padding (`int`, defaults to 1):
            The padding to use for the downsampling convolution.
        mid_block_scale_factor (`float`, defaults to 1):
            The scale factor to use for the mid block.
        act_fn (`str`, defaults to "silu"):
            The activation function to use.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups to use for the normalization. If None, normalization and activation layers is skipped
            in post-processing.
        norm_eps (`float`, defaults to 1e-5):
            The epsilon to use for the normalization.
        cross_attention_dim (`int`, defaults to 1280):
            The dimension of the cross attention features.
        transformer_layers_per_block (`int` or `Tuple[int]`, *optional*, defaults to 1):
            The number of transformer blocks of type [`~models.attention.BasicTransformerBlock`]. Only relevant for
            [`~models.unet_2d_blocks.CrossAttnDownBlock2D`], [`~models.unet_2d_blocks.CrossAttnUpBlock2D`],
            [`~models.unet_2d_blocks.UNetMidBlock2DCrossAttn`].
        encoder_hid_dim (`int`, *optional*, defaults to None):
            If `encoder_hid_dim_type` is defined, `encoder_hidden_states` will be projected from `encoder_hid_dim`
            dimension to `cross_attention_dim`.
        encoder_hid_dim_type (`str`, *optional*, defaults to `None`):
            If given, the `encoder_hidden_states` and potentially other embeddings are down-projected to text
            embeddings of dimension `cross_attention` according to `encoder_hid_dim_type`.
        attention_head_dim (`Union[int, Tuple[int]]`, defaults to 8):
            The dimension of the attention heads.
        use_linear_projection (`bool`, defaults to `False`):
        class_embed_type (`str`, *optional*, defaults to `None`):
            The type of class embedding to use which is ultimately summed with the time embeddings. Choose from None,
            `"timestep"`, `"identity"`, `"projection"`, or `"simple_projection"`.
        addition_embed_type (`str`, *optional*, defaults to `None`):
            Configures an optional embedding which will be summed with the time embeddings. Choose from `None` or
            "text". "text" will use the `TextTimeEmbedding` layer.
        num_class_embeds (`int`, *optional*, defaults to 0):
            Input dimension of the learnable embedding matrix to be projected to `time_embed_dim`, when performing
            class conditioning with `class_embed_type` equal to `None`.
        upcast_attention (`bool`, defaults to `False`):
        resnet_time_scale_shift (`str`, defaults to `"default"`):
            Time scale shift config for ResNet blocks (see `ResnetBlock2D`). Choose from `default` or `scale_shift`.
        projection_class_embeddings_input_dim (`int`, *optional*, defaults to `None`):
            The dimension of the `class_labels` input when `class_embed_type="projection"`. Required when
            `class_embed_type="projection"`.
        controlnet_conditioning_channel_order (`str`, defaults to `"rgb"`):
            The channel order of conditional image. Will convert to `rgb` if it's `bgr`.
        conditioning_embedding_out_channels (`tuple[int]`, *optional*, defaults to `(16, 32, 96, 256)`):
            The tuple of output channel for each block in the `conditioning_embedding` layer.
        global_pool_conditions (`bool`, defaults to `False`):
    """

    _skip_layers = ['conv_in', 'time_proj', 'time_embedding', 'class_embedding', 'down_blocks', 'mid_block']

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        conditioning_channels: int = 3,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        projection_class_embeddings_input_dim: Optional[int] = None,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int]] = (16, 32, 96, 256),
        global_pool_conditions: bool = False,
        addition_embed_type_num_heads: int = 64,
        lora_linear_rank: int = 4,
        lora_conv2d_rank: int = 0,
    ):
        super().__init__(
            in_channels = in_channels,
            conditioning_channels = conditioning_channels,
            flip_sin_to_cos = flip_sin_to_cos,
            freq_shift = freq_shift,
            down_block_types = down_block_types,
            only_cross_attention = only_cross_attention,
            block_out_channels = block_out_channels,
            layers_per_block = layers_per_block,
            downsample_padding = downsample_padding,
            mid_block_scale_factor = mid_block_scale_factor,
            act_fn = act_fn,
            norm_num_groups = norm_num_groups,
            norm_eps = norm_eps,
            cross_attention_dim = cross_attention_dim,
            transformer_layers_per_block = transformer_layers_per_block,
            encoder_hid_dim = encoder_hid_dim,
            encoder_hid_dim_type = encoder_hid_dim_type,
            attention_head_dim = attention_head_dim,
            num_attention_heads = num_attention_heads,
            use_linear_projection = use_linear_projection,
            class_embed_type = class_embed_type,
            addition_embed_type = addition_embed_type,
            addition_time_embed_dim = addition_time_embed_dim,
            num_class_embeds = num_class_embeds,
            upcast_attention = upcast_attention,
            resnet_time_scale_shift = resnet_time_scale_shift,
            projection_class_embeddings_input_dim = projection_class_embeddings_input_dim,
            controlnet_conditioning_channel_order = controlnet_conditioning_channel_order,
            conditioning_embedding_out_channels = conditioning_embedding_out_channels,
            global_pool_conditions = global_pool_conditions,
            addition_embed_type_num_heads = addition_embed_type_num_heads,
        )

        # Initialize lora layers
        modules = { name: layer for name, layer in self.named_modules() if name.split('.')[0] in self._skip_layers }
        for name, attn_processor in list(modules.items()):
            branches = name.split('.')
            basename = branches.pop(-1)
            parent_layer = modules.get('.'.join(branches), self)
            if isinstance(attn_processor, nn.Conv2d) and not isinstance(attn_processor, LoRACompatibleConv):
                attn_processor = LoRACompatibleConv(
                    attn_processor.in_channels,
                    attn_processor.out_channels,
                    attn_processor.kernel_size,
                    attn_processor.stride,
                    attn_processor.padding,
                    bias=False if attn_processor.bias is None else True
                )
                setattr(parent_layer, basename, attn_processor)
            if isinstance(attn_processor, nn.Linear) and not isinstance(attn_processor, LoRACompatibleLinear):
                attn_processor = LoRACompatibleLinear(
                    attn_processor.in_features,
                    attn_processor.out_features,
                    bias=False if attn_processor.bias is None else True
                )
                setattr(parent_layer, basename, attn_processor)

            if lora_conv2d_rank > 0 and isinstance(attn_processor, LoRACompatibleConv):
                in_features = attn_processor.in_channels
                out_features = attn_processor.out_channels
                kernel_size = attn_processor.kernel_size

                lora_layer = LoRAConv2dLayer(
                    in_features=in_features,
                    out_features=out_features,
                    rank=lora_linear_rank,
                    kernel_size=kernel_size,
                    stride=attn_processor.stride,
                    padding=attn_processor.padding,
                    network_alpha=None,
                )
                attn_processor.set_lora_layer(lora_layer)
            
            elif lora_linear_rank > 0 and isinstance(attn_processor, LoRACompatibleLinear):
                lora_layer = LoRALinearLayer(
                    in_features=attn_processor.in_features,
                    out_features=attn_processor.out_features,
                    rank=lora_linear_rank,
                    network_alpha=None,
                )

                # TODO: how to correct set lora layers instead of hack it (it will be set to None when enable xformer without this hack)
                original_setter = attn_processor.set_lora_layer
                attn_processor.set_lora_layer = lambda lora_layer: None if lora_layer is None else original_setter(lora_layer)

                attn_processor.set_lora_layer(lora_layer)
    
    def state_dict(self, *args, **kwargs):
        state_dict: Mapping[str, Any] = super().state_dict(*args, **kwargs)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.split('.')[0] not in self._skip_layers or '.lora_layer.' in k:
                new_state_dict[k] = v
        return new_state_dict
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        new_state_dict = OrderedDict(state_dict)
        default_state_dict = super().state_dict()
        for k, v in default_state_dict.items():
            if k.split('.')[0] in self._skip_layers and k not in new_state_dict:
                new_state_dict[k] = v
        return super().load_state_dict(new_state_dict, strict)
    
    def tie_weights(self, unet: UNet2DConditionModel):
        _tie_weights(unet.conv_in, self.conv_in)
        _tie_weights(unet.time_proj, self.time_proj)
        _tie_weights(unet.time_embedding, self.time_embedding)

        if self.class_embedding:
            _tie_weights(unet.class_embedding, self.class_embedding)
        
        _tie_weights(unet.down_blocks, self.down_blocks)
        _tie_weights(unet.mid_block, self.mid_block)

    @classmethod
    def from_unet(
        cls,
        unet: UNet2DConditionModel,
        conditioning_channels: int = 3,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int]] = (16, 32, 96, 256),
        lora_linear_rank: int = 4,
        lora_conv2d_rank: int = 0,
    ):
        r"""
        Instantiate a [`ControlNetModel`] from [`UNet2DConditionModel`].

        Parameters:
            unet (`UNet2DConditionModel`):
                The UNet model weights to copy to the [`ControlNetModel`]. All configuration options are also copied
                where applicable.
        """
        transformer_layers_per_block = (
            unet.config.transformer_layers_per_block if "transformer_layers_per_block" in unet.config else 1
        )
        encoder_hid_dim = unet.config.encoder_hid_dim if "encoder_hid_dim" in unet.config else None
        encoder_hid_dim_type = unet.config.encoder_hid_dim_type if "encoder_hid_dim_type" in unet.config else None
        addition_embed_type = unet.config.addition_embed_type if "addition_embed_type" in unet.config else None
        addition_time_embed_dim = (
            unet.config.addition_time_embed_dim if "addition_time_embed_dim" in unet.config else None
        )

        controllora: ControlLoRAModel = cls(
            conditioning_channels=conditioning_channels,
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=unet.config.in_channels,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            freq_shift=unet.config.freq_shift,
            down_block_types=unet.config.down_block_types,
            only_cross_attention=unet.config.only_cross_attention,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            downsample_padding=unet.config.downsample_padding,
            mid_block_scale_factor=unet.config.mid_block_scale_factor,
            act_fn=unet.config.act_fn,
            norm_num_groups=unet.config.norm_num_groups,
            norm_eps=unet.config.norm_eps,
            cross_attention_dim=unet.config.cross_attention_dim,
            attention_head_dim=unet.config.attention_head_dim,
            num_attention_heads=unet.config.num_attention_heads,
            use_linear_projection=unet.config.use_linear_projection,
            class_embed_type=unet.config.class_embed_type,
            num_class_embeds=unet.config.num_class_embeds,
            upcast_attention=unet.config.upcast_attention,
            resnet_time_scale_shift=unet.config.resnet_time_scale_shift,
            projection_class_embeddings_input_dim=unet.config.projection_class_embeddings_input_dim,
            controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
            lora_linear_rank=lora_linear_rank,
            lora_conv2d_rank=lora_conv2d_rank,
        )

        controllora.tie_weights(unet)

        return controllora
