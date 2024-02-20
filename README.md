# ControlLoRA Version 2: A Lightweight Neural Network To Control Stable Diffusion Spatial Information Version 2

ControlLoRA Version 2 is a neural network structure extended from Controlnet to control diffusion models by adding extra conditions.

ControlLoRA Version 2 uses the same structure as Controlnet. But its core weight comes from UNet, unmodified. Only hint image encoding layers, linear lora layers and conv2d lora layers used for weight offset are trained.

The main idea is from my [ControlLoRA](https://github.com/HighCWu/ControlLoRA) and sdxl [control-lora](https://huggingface.co/stabilityai/control-lora).

The current implementation idea is basically the same as sdxl's control-lora. But I mainly extend the ControlNetModel implementation from diffusers and use the user-friendly sd v1.5 for training. (The training method of sdxl control-lora could be easily used after modification according to [train_controlnet_sdxl.py](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet_sdxl.py). My training code is also from [train_controlnet.py](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet.py))

Notice: I didn't train some extra layers like the norm layers (while stabilityai did). So this repo is more appropriate to the name of control-lora.

## Why Version 2

My first version of [ControlLoRA](https://github.com/HighCWu/ControlLoRA) directly used the spatial information after simple convolutions to perform LoRA offset on the linear layers in the attention layers. 
This results in a lot of prior information in UNet not being utilized. 

Therefore, in version 2, I directly operate ControlNet, but do not change the weights of some UNet layers used in ControlNet, and only train the hint image encoder and the LoRA offset weights to achieve lightweighting. 

Thanks to [Locon](https://github.com/KohakuBlueleaf/LyCORIS)'s idea, we could also apply LoRA offset to the conv2d weights.

## Training

See scripts in the `scripts` folder

## Available Pretrained Models (WIP)

[sd-controllora-face-landmarks](https://huggingface.co/HighCWu/sd-controllora-face-landmarks)

## Example

1. Clone ControlLoRA from [Github](https://github.com/HighCWu/control-lora-v2):
```sh
$ git clone https://github.com/HighCWu/control-lora-v2
```

2. Enter the repo dir:
```sh
$ cd control-lora-v2
```

3. Run code:
```py
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, UNet2DConditionModel, UniPCMultistepScheduler
from models.control_lora import ControlLoRAModel

image = Image.open("./docs/imgs/face_landmarks1.jpeg")

base_model = "runwayml/stable-diffusion-v1-5"

unet = UNet2DConditionModel.from_pretrained(
    base_model, subfolder="unet", torch_dtype=torch.float16
)
controllora: ControlLoRAModel = ControlLoRAModel.from_pretrained(
    "HighCWu/sd-controllora-face-landmarks", torch_dtype=torch.float16
)
controllora.tie_weights(unet)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model, unet=unet, controlnet=controllora, safety_checker=None, torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

image = pipe("Girl smiling, professional dslr photograph, high quality", image, num_inference_steps=20).images[0]

image.show()
```

Replace the hint image and model paths according to your needs. For example, you could apply controllora to anime by replace `runwayml/stable-diffusion-v1-5` to `ckpt/anything-v3-vae-swapped`.

## Discuss together

QQ Group: [艾梦的小群](https://jq.qq.com/?_wv=1027&k=yMtGIF1Q)

QQ Channel: [艾梦的AI造梦堂](https://pd.qq.com/s/1qyek3j0e)

Discord: [AI Players - AI Dream Bakery](https://discord.gg/zcJszfPrZs)

## Citation

    @software{wu2023controllorav2,
        author = {Wu Hecong},
        month = {9},
        title = {{ControlLoRA Version 2: A Lightweight Neural Network To Control Stable Diffusion Spatial Information Version 2}},
        url = {https://github.com/HighCWu/control-lora-2},
        version = {1.0.0},
        year = {2023}
    }
