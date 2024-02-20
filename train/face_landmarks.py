from trainer import main
from train.config.face_landmarks import config

config.output_dir = "output/sd-control-lora-face-landmarks"
config.tracker_project_name = "control-lora"

if __name__ == '__main__':
    main(config)
