#!/bin/bash

# Lambda Calculus Geometric Training Test
# 3 epochs × 50 steps = 150 total steps of revolutionary geometric training

python run.py \
  --run_name "lambda_geometric_test" \
  --dataset_name "Creekside/GRPO-Lambda-ParsedForUnsloth" \
  --dataset_source "huggingface" \
  --dataset_config_name "default" \
  --epochs 3 \
  --max_batches_per_epoch 50 \
  --training_mode "geometric" \
  --geometric_enabled \
  --batch_size 4 \
  --eval_every 25 \
  --geometric_rotation_dimensions 6 \
  --geometric_learning_rate 0.001 \
  --geometric_expert_learning_rate 0.0001 \
  --geometric_lambda_cognitive_rotations
