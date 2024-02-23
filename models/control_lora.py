from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from diffusers.configuration_utils import register_to_config
from diffusers.utils import logging
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.models.controlnet import ControlNetModel, ControlNetOutput
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


class DoRAConv2dLayer(LoRAConv2dLayer):
    r"""
    A convolutional layer that is used with DoRA.

    Parameters:
        in_features (`int`):
            Number of input features.
        out_features (`int`):
            Number of output features.
        rank (`int`, `optional`, defaults to 4):
            The rank of the LoRA layer.
        kernel_size (`int` or `tuple` of two `int`, `optional`, defaults to 1):
            The kernel size of the convolution.
        stride (`int` or `tuple` of two `int`, `optional`, defaults to 1):
            The stride of the convolution.
        padding (`int` or `tuple` of two `int` or `str`, `optional`, defaults to 0):
            The padding of the convolution.
        network_alpha (`float`, `optional`, defaults to `None`):
            The value of the network alpha used for stable learning and preventing underflow. This value has the same
            meaning as the `--network_alpha` option in the kohya-ss trainer script. See
            https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        kernel_size: Union[int, Tuple[int, int]] = (1, 1),
        stride: Union[int, Tuple[int, int]] = (1, 1),
        padding: Union[int, Tuple[int, int], str] = 0,
        network_alpha: Optional[float] = None,
    ):
        super().__init__(
            in_features,
            out_features,
            rank=rank,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            network_alpha=network_alpha
        )

        self.magnitude = nn.Parameter(torch.ones(1, in_features, *self.down.kernel_size))
        self.magnitude_initialized = False
        self.register_buffer('magnitude_initialized_buf', torch.tensor(False))

    def forward(self, w_orig: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        if self.up.weight.device != w_orig.device or self.up.weight.dtype != w_orig.dtype:
            self.to(w_orig)
        
        if not self.magnitude_initialized:
            if not self.magnitude_initialized_buf.item():
                w_norm: torch.Tensor = w_orig.flatten(start_dim=1).norm(p=2, dim=0, keepdim=True)
                self.magnitude.data = w_norm.view_as(self.magnitude)
                self.magnitude_initialized_buf.data = ~self.magnitude_initialized_buf
            self.magnitude_initialized = True
        
        w_up = self.up.weight
        w_down = self.down.weight

        if self.network_alpha is not None:
            w_up = w_up * self.network_alpha / self.rank
        
        if w_up.device == torch.device('cpu') and (w_up.dtype == torch.float16 or w_up.dtype == torch.bfloat16):
            w_up = w_up.float()
            w_down = w_down.float()

        lora = torch.mm(w_up.flatten(start_dim=1), w_down.flatten(start_dim=1))
        adapted = w_orig.flatten(start_dim=1) + lora

        weight_norm: torch.Tensor = adapted.norm(p=2, dim=0, keepdim=True)
        norm_lora = lora / weight_norm.detach()

        return scale * self.magnitude * norm_lora.view_as(w_orig)


class DoRACompatibleConv(LoRACompatibleConv):
    """
    A convolutional layer that can be used with DoRA.
    """
    lora_layer: DoRAConv2dLayer

    def _fuse_lora(self, lora_scale: float = 1.0, safe_fusing: bool = False):
        if self.lora_layer is None:
            return

        dtype, device = self.weight.data.dtype, self.weight.data.device

        w_orig = self.weight.data.float()
        w_up = self.lora_layer.up.weight.data.float()
        w_down = self.lora_layer.down.weight.data.float()
        w_magnitude = self.lora_layer.magnitude.data.float()

        if self.lora_layer.network_alpha is not None:
            w_up = w_up * self.lora_layer.network_alpha / self.lora_layer.rank

        lora = torch.mm(w_up.flatten(start_dim=1), w_down.flatten(start_dim=1))
        adapted = w_orig.flatten(start_dim=1) + lora

        weight_norm: torch.Tensor = adapted.norm(p=2, dim=0, keepdim=True)
        norm_lora = lora / weight_norm.detach()
        fused_weight = w_orig + lora_scale * w_magnitude * norm_lora.view_as(w_orig)

        if safe_fusing and torch.isnan(fused_weight).any().item():
            raise ValueError(
                "This DoRA weight seems to be broken. "
                f"Encountered NaN values when trying to fuse DoRA weights for {self}."
                "DoRA weights will not be fused."
            )

        self.weight.data = fused_weight.to(device=device, dtype=dtype)

        # we can drop the lora layer now
        self.lora_layer = None

        # offload the up and down matrices to CPU to not blow the memory
        self.w_up = w_up.cpu()
        self.w_down = w_down.cpu()
        self.w_magnitude = w_magnitude.cpu()
        self.w_norm_lora = norm_lora.cpu()
        self._lora_scale = lora_scale

    def _unfuse_lora(self):
        if not (getattr(self, "w_up", None) is not None and getattr(self, "w_down", None) is not None and 
                getattr(self, "w_magnitude", None) is not None and getattr(self, "w_norm_lora", None) is not None):
            return

        fused_weight = self.weight.data
        dtype, device = fused_weight.data.dtype, fused_weight.data.device

        w_magnitude: torch.Tensor = self.w_magnitude.to(device).float()
        w_norm_lora: torch.Tensor = self.w_norm_lora.to(device).float()

        unfused_weight = self.weight - self._lora_scale * w_magnitude * w_norm_lora.view_as(self.weight)
        self.weight.data = unfused_weight.to(device=device, dtype=dtype)

        self.w_up = None
        self.w_down = None
        self.w_magnitude = None
        self.w_norm_lora = None

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        if self.weight.device != hidden_states.device:
            self.to(hidden_states.device)
        weight = self.weight if self.lora_layer is None else self.weight + scale * self.lora_layer(self.weight)

        return F.conv2d(
            hidden_states, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class DoRALinearLayer(LoRALinearLayer):
    r"""
    A linear layer that is used with DoRA.

    Parameters:
        in_features (`int`):
            Number of input features.
        out_features (`int`):
            Number of output features.
        rank (`int`, `optional`, defaults to 4):
            The rank of the LoRA layer.
        network_alpha (`float`, `optional`, defaults to `None`):
            The value of the network alpha used for stable learning and preventing underflow. This value has the same
            meaning as the `--network_alpha` option in the kohya-ss trainer script. See
            https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        device (`torch.device`, `optional`, defaults to `None`):
            The device to use for the layer's weights.
        dtype (`torch.dtype`, `optional`, defaults to `None`):
            The dtype to use for the layer's weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            in_features,
            out_features,
            rank=rank,
            network_alpha=network_alpha,
            device=device,
            dtype=dtype
        )

        self.magnitude = nn.Parameter(torch.ones(1, in_features))
        self.magnitude_initialized = False
        self.register_buffer('magnitude_initialized_buf', torch.tensor(False))

    def forward(self, w_orig: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        if self.up.weight.device != w_orig.device or self.up.weight.dtype != w_orig.dtype:
            self.to(w_orig)
        
        if not self.magnitude_initialized:
            if not self.magnitude_initialized_buf.item():
                w_norm: torch.Tensor = w_orig.norm(p=2, dim=0, keepdim=True)
                self.magnitude.data = w_norm
                self.magnitude_initialized_buf.data = ~self.magnitude_initialized_buf
            self.magnitude_initialized = True

        w_up = self.up.weight
        w_down = self.down.weight

        if self.network_alpha is not None:
            w_up = w_up * self.network_alpha / self.rank
        
        if w_up.device == torch.device('cpu') and (w_up.dtype == torch.float16 or w_up.dtype == torch.bfloat16):
            w_up = w_up.float()
            w_down = w_down.float()

        lora = torch.mm(w_up, w_down).to(w_orig)
        adapted = w_orig + lora

        weight_norm: torch.Tensor = adapted.norm(p=2, dim=0, keepdim=True)
        norm_lora = lora / weight_norm.detach()

        return scale * self.magnitude * norm_lora


class DoRACompatibleLinear(LoRACompatibleLinear):
    """
    A Linear layer that can be used with DoRA.
    """
    lora_layer: DoRALinearLayer

    def _fuse_lora(self, lora_scale: float = 1.0, safe_fusing: bool = False):
        if self.lora_layer is None:
            return

        dtype, device = self.weight.data.dtype, self.weight.data.device

        w_orig = self.weight.data.float()
        w_up = self.lora_layer.up.weight.data.float()
        w_down = self.lora_layer.down.weight.data.float()
        w_magnitude = self.lora_layer.magnitude.data.float()

        if self.lora_layer.network_alpha is not None:
            w_up = w_up * self.lora_layer.network_alpha / self.lora_layer.rank

        lora = torch.mm(w_up, w_down)
        adapted = w_orig + lora

        weight_norm: torch.Tensor = adapted.norm(p=2, dim=0, keepdim=True)
        norm_lora = lora / weight_norm.detach()
        fused_weight = w_orig + lora_scale * w_magnitude * norm_lora

        if safe_fusing and torch.isnan(fused_weight).any().item():
            raise ValueError(
                "This DoRA weight seems to be broken. "
                f"Encountered NaN values when trying to fuse DoRA weights for {self}."
                "DoRA weights will not be fused."
            )

        self.weight.data = fused_weight.to(device=device, dtype=dtype)

        # we can drop the lora layer now
        self.lora_layer = None

        # offload the up and down matrices to CPU to not blow the memory
        self.w_up = w_up.cpu()
        self.w_down = w_down.cpu()
        self.w_magnitude = w_magnitude.cpu()
        self.w_norm_lora = norm_lora.cpu()
        self._lora_scale = lora_scale

    def _unfuse_lora(self):
        if not (getattr(self, "w_up", None) is not None and getattr(self, "w_down", None) is not None and 
                getattr(self, "w_magnitude", None) is not None and getattr(self, "w_norm_lora", None) is not None):
            return

        fused_weight = self.weight.data
        dtype, device = fused_weight.data.dtype, fused_weight.data.device

        w_magnitude: torch.Tensor = self.w_magnitude.to(device).float()
        w_norm_lora: torch.Tensor = self.w_norm_lora.to(device).float()

        unfused_weight = self.weight - self._lora_scale * w_magnitude * w_norm_lora
        self.weight.data = unfused_weight.to(device=device, dtype=dtype)

        self.w_up = None
        self.w_down = None
        self.w_magnitude = None
        self.w_norm_lora = None

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        if self.weight.device != hidden_states.device:
            self.to(hidden_states.device)
        weight = self.weight if self.lora_layer is None else self.weight + scale * self.lora_layer(self.weight)

        return F.linear(hidden_states, weight, self.bias)


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
            TODO(Patrick) - unused parameter.
        addition_embed_type_num_heads (`int`, defaults to 64):
            The number of heads to use for the `TextTimeEmbedding` layer.
    """

    _skip_layers = ['conv_in', 'time_proj', 'time_embedding', 'class_embedding', 'down_blocks', 'mid_block', 'vae']

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        conditioning_channels: int = 3,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int, ...]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        projection_class_embeddings_input_dim: Optional[int] = None,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
        global_pool_conditions: bool = False,
        addition_embed_type_num_heads: int = 64,
        lora_linear_rank: int = 4,
        lora_conv2d_rank: int = 0,
        use_conditioning_latent: bool = False,
        use_same_level_conditioning_latent: bool = False,
        use_dora: bool = False,
    ):
        if use_conditioning_latent:
            conditioning_channels = in_channels

        super().__init__(
            in_channels = in_channels,
            conditioning_channels = conditioning_channels,
            flip_sin_to_cos = flip_sin_to_cos,
            freq_shift = freq_shift,
            down_block_types = down_block_types,
            mid_block_type = mid_block_type,
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

        vae: AutoencoderKL = None
        self.vae = vae
        self.use_conditioning_latent = use_conditioning_latent
        self.use_same_level_conditioning_latent = use_same_level_conditioning_latent
        if use_same_level_conditioning_latent:
            # Use latent as cond
            del self.controlnet_cond_embedding
            self.controlnet_cond_embedding = lambda _: 0
            self.config['use_conditioning_latent'] = False

        conv_cls = DoRACompatibleConv if use_dora else LoRACompatibleConv
        conv_lora_cls = DoRAConv2dLayer if use_dora else LoRAConv2dLayer
        linear_cls = DoRACompatibleLinear if use_dora else LoRACompatibleLinear
        linear_lora_cls = DoRALinearLayer if use_dora else LoRALinearLayer

        # Initialize lora layers
        modules = { name: layer for name, layer in self.named_modules() if name.split('.')[0] in self._skip_layers }
        for name, attn_processor in list(modules.items()):
            branches = name.split('.')
            basename = branches.pop(-1)
            parent_layer = modules.get('.'.join(branches), self)
            if isinstance(attn_processor, nn.Conv2d):
                attn_processor = conv_cls(
                    attn_processor.in_channels,
                    attn_processor.out_channels,
                    attn_processor.kernel_size,
                    attn_processor.stride,
                    attn_processor.padding,
                    bias=False if attn_processor.bias is None else True
                )
                setattr(parent_layer, basename, attn_processor)
            if isinstance(attn_processor, nn.Linear):
                attn_processor = linear_cls(
                    attn_processor.in_features,
                    attn_processor.out_features,
                    bias=False if attn_processor.bias is None else True
                )
                setattr(parent_layer, basename, attn_processor)

            if lora_conv2d_rank > 0 and isinstance(attn_processor, conv_cls):
                in_features = attn_processor.in_channels
                out_features = attn_processor.out_channels
                kernel_size = attn_processor.kernel_size

                lora_layer = conv_lora_cls(
                    in_features=in_features,
                    out_features=out_features,
                    rank=lora_linear_rank,
                    kernel_size=kernel_size,
                    stride=attn_processor.stride,
                    padding=attn_processor.padding,
                    network_alpha=None,
                )
                attn_processor.set_lora_layer(lora_layer)
            
            elif lora_linear_rank > 0 and isinstance(attn_processor, linear_cls):
                lora_layer = linear_lora_cls(
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

    def bind_vae(self, vae: AutoencoderKL):
        self.vae = vae

    @classmethod
    def from_unet(
        cls,
        unet: UNet2DConditionModel,
        conditioning_channels: int = 3,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int]] = (16, 32, 96, 256),
        lora_linear_rank: int = 4,
        lora_conv2d_rank: int = 0,
        use_conditioning_latent: bool = False,
        use_same_level_conditioning_latent: bool = False,
        use_dora: bool = False,
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
            use_conditioning_latent=use_conditioning_latent,
            use_same_level_conditioning_latent=use_same_level_conditioning_latent,
            use_dora=use_dora,
        )

        controllora.tie_weights(unet)

        return controllora

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.FloatTensor,
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple[Tuple[torch.FloatTensor, ...], torch.FloatTensor]]:
        if self.use_conditioning_latent or self.use_same_level_conditioning_latent:
            with torch.no_grad():
                controlnet_cond = controlnet_cond * 2 - 1
                controlnet_cond = self.vae.encode(controlnet_cond.to(self.vae.device, self.vae.dtype)).latent_dist.sample()
                controlnet_cond = controlnet_cond.to(sample) * self.vae.config.scaling_factor
                if self.use_conditioning_latent:
                    vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
                    controlnet_cond = F.interpolate(controlnet_cond, scale_factor=vae_scale_factor, mode='nearest')
        if self.use_same_level_conditioning_latent:
            sample = controlnet_cond
        return super().forward(
            sample,
            timestep,
            encoder_hidden_states,
            controlnet_cond,
            conditioning_scale,
            class_labels = class_labels,
            timestep_cond = timestep_cond,
            attention_mask = attention_mask,
            added_cond_kwargs = added_cond_kwargs,
            cross_attention_kwargs = cross_attention_kwargs,
            guess_mode = guess_mode,
            return_dict = return_dict
        )
