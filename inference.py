from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, UNet2DConditionModel, UniPCMultistepScheduler
import torch
from PIL import Image
from models.controllora import ControlLoRAModel

image = Image.open("./docs/imgs/face_landmarks1.jpeg")

unet = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype=torch.float16, cache_dir='.cache'
)
controllora: ControlLoRAModel = ControlLoRAModel.from_pretrained(
    "HighCWu/sd-controllora-face-landmarks", torch_dtype=torch.float16, cache_dir='.cache'
)
controllora.tie_weights(unet)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", unet=unet, controlnet=controllora, safety_checker=None, torch_dtype=torch.float16, cache_dir='.cache'
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

image = pipe("Girl smiling, professional dslr photograph, high quality", image, num_inference_steps=20).images[0]

image.show()
