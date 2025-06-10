#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=4

# English

# models=( "osatlas-7b" "osatlas-4b" "uground" "ugroundv1"  )
models=( "uground" )
dataset="pro"

for model in "${models[@]}" 
do
    python eval_screenspot_pro.py  \
        --model_type ${model}  \
        --screenspot_imgs "./data/${dataset}/images"  \
        --screenspot_test "./data/${dataset}/annotations"  \
        --task "all" \
        --language "en" \
        --gt_type "positive" \
        --log_path "./results/${dataset}/${model}.json" \
        --inst_style "instruction" \
        --max_iter 5 \
        --threshold 0.05 \

done