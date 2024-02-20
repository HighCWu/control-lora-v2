import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, UNet2DConditionModel, UniPCMultistepScheduler
from models.control_lora import ControlLoRAModel

image = Image.open("./docs/imgs/face_landmarks1.jpeg")

base_model = "ckpt/anything-v3-vae-swapped"

unet = UNet2DConditionModel.from_pretrained(
    base_model, subfolder="unet", torch_dtype=torch.float16, cache_dir='.cache'
)
control_lora: ControlLoRAModel = ControlLoRAModel.from_pretrained(
    "HighCWu/sd-control-lora-face-landmarks", torch_dtype=torch.float16, cache_dir='.cache'
)
control_lora.tie_weights(unet)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model, unet=unet, controlnet=control_lora, safety_checker=None, torch_dtype=torch.float16, cache_dir='.cache'
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

image = pipe(
    "masterpiece, best quality, high quality, Girl smiling", 
    image, 
    negative_prompt="lowres, bad anatomy, text, error, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    num_inference_steps=20).images[0]

image.show()
