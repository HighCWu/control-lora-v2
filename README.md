# ControlLoRA Version 2: A Lightweight Neural Network To Control Stable Diffusion Spatial Information Version 2

🎉 2024.7.31： ControlLoRA Version 3 is available in [control-lora-3](https://github.com/HighCWu/control-lora-v3).

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

Run this command:
```sh
$ accelerate launch -m train.<TASK>
```
\<TASK\> means the module in the `train` folder. For example, you could run this command to launch the train script `train/tutorial.py`:
```sh
$ accelerate launch -m train.tutorial
```

Run a command with `--push_to_hub` to release your model to huggingface hub after training:
```sh
$ accelerate launch -m train.<TASK> --push_to_hub
```

## Available Pretrained Models (WIP)

[sd-control-lora-face-landmarks](https://huggingface.co/HighCWu/sd-control-lora-face-landmarks)

[sd-control-lora-head3d](https://huggingface.co/HighCWu/sd-control-lora-head3d)

[sd-latent-control-dora-rank128-head3d](https://huggingface.co/HighCWu/sd-latent-control-dora-rank128-head3d)

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
```sh
$ python -m infer.<TASK>
```

Replace \<TASK\> to the task module. For example:
```sh
$ python -m infer.face_landmarks
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
