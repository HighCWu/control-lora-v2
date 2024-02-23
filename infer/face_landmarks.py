import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from huggingface_hub import whoami
from diffusers import StableDiffusionControlNetPipeline, UNet2DConditionModel, UniPCMultistepScheduler
from models.control_lora import ControlLoRAModel
from train.face_landmarks import config

seed = 46

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

image = Image.open("./docs/imgs/face_landmarks1.jpeg")

base_model = "ckpt/anything-v3-vae-swapped"

unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
    base_model, subfolder="unet", torch_dtype=dtype, cache_dir='.cache'
)
pretrained_model_name_or_path = config.output_dir
if len(glob.glob(f'{config.output_dir}/*.safetensors')) == 0:
    # Get the most recent checkpoint
    dirs = os.listdir(config.output_dir) if os.path.exists(config.output_dir) else []
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    if len(dirs) > 0:
        pretrained_model_name_or_path = os.path.join(config.output_dir, dirs[-1])
    else:
        user_info = whoami(token=config.hub_token)
        username = user_info.get('name', None)
        if username is not None:
            pretrained_model_name_or_path = f"{username}/{config.tracker_project_name}"

try:
    control_lora: ControlLoRAModel = ControlLoRAModel.from_pretrained(
        pretrained_model_name_or_path, torch_dtype=dtype, cache_dir='.cache', token=config.hub_token
    )
except:
    control_lora: ControlLoRAModel = ControlLoRAModel.from_pretrained(
        f"HighCWu/{config.tracker_project_name}", torch_dtype=dtype, cache_dir='.cache'
    )

control_lora.tie_weights(unet)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model, unet=unet, controlnet=control_lora, safety_checker=None, torch_dtype=dtype, cache_dir='.cache'
).to(device)
control_lora.bind_vae(pipe.vae)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
pipe.enable_xformers_memory_efficient_attention()

# pipe.enable_model_cpu_offload()

generator = torch.Generator(pipe.device).manual_seed(seed)
image_out = pipe(
    "masterpiece, best quality, high quality, Girl smiling", 
    image, 
    generator=generator,
    negative_prompt="lowres, bad anatomy, text, error, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    num_inference_steps=20).images[0]

# show hint and output image
images = [image, image_out]
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)

image_cat = Image.new('RGB', (total_width, max_height))

x_offset = 0
for img in images:
    image_cat.paste(img, (x_offset,0))
    x_offset += img.size[0]

image_cat.show()
