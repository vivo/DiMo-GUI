#!/bin/bash
set -e

# 指定可用的 GPU 编号，例如 0 或 0,1
export CUDA_VISIBLE_DEVICES=4

# English

# models=("cogagent24" "ariaui" "uground" "osatlas-7b" "osatlas-4b" "showui" "seeclick" "qwen1vl" "qwen2vl" "minicpmv" "cogagent" "gpt4o" "fuyu" "internvl" "ugroundv1" )
models=( "osatlas-7b" )

for model in "${models[@]}" 
do
    python eval_screenspot_pro.py  \
        --model_type ${model}  \
        --screenspot_imgs "./data/v2/images"  \
        --screenspot_test "./data/v2/annotations"  \
        --task "all" \
        --language "en" \
        --gt_type "positive" \
        --log_path "./results/v2/${model}.json" \
        --inst_style "instruction" \
        --max_iter 3 \
        --threshold 0.05 \

done