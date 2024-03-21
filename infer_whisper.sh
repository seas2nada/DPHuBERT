#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

. tools/activate_python.sh

set -e
set -u
set -o pipefail

exp_dir=$PWD/exp/whisper-medium
model_name=$PWD/pretrained/whisper-medium.hf.pth
language="ko"

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --exp_dir) exp_dir="$2"; shift ;;
        --model_name) model_name="$2"; shift ;;
        --language) language="$2"; shift ;;
        --whisper_model_name) whisper_model_name="$2"; shift ;;
        --eval_metric) eval_metric="$2"; shift ;;
        --dataset) dataset="$2"; shift ;;
        *) ;;
    esac
    shift
done

python whisper/whisper_decoding.py \
    --model_name ${model_name} \
    --language $language \
    --whisper_model_name ${whisper_model_name} \
    --exp_dir ${exp_dir} \
    --eval_metric ${eval_metric} \
    --dataset ${dataset} \
    --batch_size 16 2>&1 | tee ${exp_dir}/infer-${language}.log || exit 1;
