#!/bin/bash

python main_eurosat.py \
  --data_dir datasets/eurosat \
  --backbone_type checkpoint --ckpt_path checkpoints/seco_resnet50_1m.ckpt
