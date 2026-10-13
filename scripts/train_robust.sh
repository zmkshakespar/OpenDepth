#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${MODEL_DIR:?Set MODEL_DIR to the Step1X-Edit v1.0 model directory}"
: "${DATA_CONFIG:?Set DATA_CONFIG to your dataset TOML file}"
NUM_GPUS="${NUM_GPUS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/training}"
export DISABLE_LIGER=1
export TORCHDYNAMO_DISABLE=1

accelerate launch --num_processes "$NUM_GPUS" --mixed_precision bf16 train.py \
  --pretrained_model_name_or_path "$MODEL_DIR/step1x-edit-i1258.safetensors" \
  --qwen2p5vl "$MODEL_DIR/Qwen2.5-VL-7B-Instruct" --ae "$MODEL_DIR/vae.safetensors" \
  --dataset_config "$DATA_CONFIG" --output_dir "$OUTPUT_DIR" --output_name opendepth \
  --network_module library.lora_module --network_dim 64 --network_alpha 32 \
  --network_train_unet_only --optimizer_type adamw8bit --learning_rate 1e-4 \
  --max_grad_norm 1.0 --max_train_epochs 20 --save_every_n_epochs 1 \
  --mixed_precision bf16 --save_precision bf16 --gradient_accumulation_steps 2 \
  --mix_dataset --dataset_probs "0.8,0.1,0.1" \
  --gradient_checkpointing --highvram --fp8_base --cache_latents_to_disk \
  --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk \
  --timestep_sampling shift --discrete_flow_shift 3.1582 --model_prediction_type raw \
  --guidance_scale 1.0 --discriminative_mode --discriminative_timestep 1.0 \
  --enable_perturbation --perturbation_prob 1.0 --perturbation_weight 1.0 \
  --perturb_severity_min 1 --perturb_severity_max 5 --perturbation_workers 4 \
  --feature_loss_layers 4 --feature_loss_weight 1.0 --gt_perturbed_weight 1.0 \
  --enable_rgb_reconstruction --rgb_self_weight 1.0 --rgb_cross_weight 1.0 \
  --max_data_loader_n_workers 2 --seed 42 --save_model_as safetensors "$@"
