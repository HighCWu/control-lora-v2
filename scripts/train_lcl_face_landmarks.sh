cd $(dirname "$0")/..

accelerate launch trainer.py \
 --cache_dir=".cache" \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --output_dir="output/sd-latent-control-lora-face-landmarks" \
 --dataset_name="multimodalart/facesyntheticsspigacaptioned" \
 --conditioning_image_column=spiga_seg \
 --image_column=image \
 --caption_column=image_caption \
 --resolution=512 \
 --control_lora_linear_rank=32 \
 --control_lora_conv2d_rank=32 \
 --use_conditioning_latent \
 --learning_rate=1e-4 \
 --validation_image "./docs/imgs/face_landmarks1.jpeg" "./docs/imgs/face_landmarks2.jpeg" "./docs/imgs/face_landmarks3.jpeg" \
 --validation_prompt "High-quality close-up dslr photo of man wearing a hat with trees in the background" "Girl smiling, professional dslr photograph, dark background, studio lights, high quality" "Portrait of a clown face, oil on canvas, bittersweet expression" \
 --train_batch_size=4 \
 --max_train_steps=75000 \
 --conditioning_type_name=face-landmarks \
 --tracker_project_name="latent-control-lora" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=5000 \
 --validation_steps=5000 \
 --report_to wandb \
 --resume_from_checkpoint latest \
 # --push_to_hub
