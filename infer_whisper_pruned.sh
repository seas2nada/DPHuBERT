#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

. tools/activate_python.sh

set -e
set -u
set -o pipefail

exp_dir=$PWD/exp/whisper_trnone_8gpu
model_name=$PWD/exp/whisper_trnone_8gpu/ckpts/pruned_hubert_base.pth
language="ko"

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --exp_dir) exp_dir="$2"; shift ;;
        --model_name) model_name="$2"; shift ;;
        --language) language="$2"; shift ;;
        *) ;;
    esac
    shift
done

python whisper/whisper_decoding.py \
    --model_name ${model_name} \
    --language $language \
    --whisper_model_name "openai/whisper-medium" \
    --exp_dir ${exp_dir} \
    --batch_size 16 2>&1 | tee ${exp_dir}/infer-${language}.log || exit 1;