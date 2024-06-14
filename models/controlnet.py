from diffusers.models.unets.unet_2d_condition import (
    UNet2DConditionModel,
    UNet2DConditionLoadersMixin,
)
from diffusers.models.controlnet import ControlNetModel as ControlNetModelOriginal
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.pipelines.controlnet.pipeline_controlnet import (
    StableDiffusionControlNetPipeline,
)
from diffusers.loaders.peft import PeftAdapterMixin


class ControlNetModel(
    ControlNetModelOriginal, PeftAdapterMixin, UNet2DConditionLoadersMixin
): ...


if __name__ == "__main__":
    # example
    from peft import LoraConfig
    from peft.tuners.lora.layer import Linear

    print("")
    print("")
    print("Test adapter and saver:")
    controlnet_lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    # controlnet = ControlNetModel.from_unet(unet)
    controlnet: ControlNetModel = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", cache_dir=".cache"
    )
    controlnet.add_adapter(controlnet_lora_config)

    for n, m in controlnet.named_modules():
        if isinstance(m, Linear):
            print(m)
            print("Found peft lora linear:", n, type(m))
    print(type(controlnet))

    controlnet.save_attn_procs("tmp_controlnet_lora_dir")

    print("")
    print("")
    print("Test loader:")
    controlnet2: ControlNetModel = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", cache_dir=".cache"
    )
    controlnet2.load_attn_procs("tmp_controlnet_lora_dir")

    for n, m in controlnet2.named_modules():
        if isinstance(m, Linear):
            print(m)
            print("Found peft lora linear:", n, type(m))
    print(type(controlnet2))
