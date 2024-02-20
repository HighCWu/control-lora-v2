from trainer import main
from train.config.face_landmarks import config

config.output_dir = "output/sd-latent-control-lora-face-landmarks"
config.tracker_project_name = "latent-control-lora"
config.use_conditioning_latent = True

if __name__ == '__main__':
    main(config)
