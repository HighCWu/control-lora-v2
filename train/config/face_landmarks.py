from trainer import parse_args

config = parse_args()
config.cache_dir = ".cache"
config.pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
config.dataset_name = "multimodalart/facesyntheticsspigacaptioned"
config.conditioning_image_column = "spiga_seg"
config.image_column = "image"
config.caption_column = "image_caption"
config.resolution = 512
config.control_lora_linear_rank = 32
config.control_lora_conv2d_rank = 32
config.learning_rate = 1e-4
config.validation_image = ["./docs/imgs/face_landmarks1.jpeg", "./docs/imgs/face_landmarks2.jpeg", "./docs/imgs/face_landmarks3.jpeg"]
config.validation_prompt = ["High-quality close-up dslr photo of man wearing a hat with trees in the background", "Girl smiling, professional dslr photograph, dark background, studio lights, high quality", "Portrait of a clown face, oil on canvas, bittersweet expression"]
config.train_batch_size = 4
config.max_train_steps = 75000
config.conditioning_type_name = "face-landmarks"
config.enable_xformers_memory_efficient_attention = True
config.checkpointing_steps = 5000
config.validation_steps = 5000
config.report_to = "wandb"
config.resume_from_checkpoint = "latest"
# config.push_to_hub = True
