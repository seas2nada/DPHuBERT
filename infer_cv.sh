#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

. tools/activate_python.sh

set -e
set -u
set -o pipefail

model_name=$PWD/exp/wav2vec2-base_TED100h_sp0.20_spup5000_lr0.0002_up15000_max50000_layer2layer0.4,8,12_distill_weight1.0_reglr0.02_conv,head,interm_ctc0.0005_mask0.2_chanmask0.2/ckpts/pruned_hubert_base.pth

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --model_name) model_name="$2"; shift ;;
        *) ;;
    esac
    shift
done

for subset in "cv-dev" "cv-test"; do
    data_dir=$PWD/data/commonvoice/$subset/

    inference_result=$PWD/inference_result/
    wordscore=-1
    lmweight=2
    silscore=0
    num_gpus=1

    result_path=$(echo $model_name | sed 's#/ckpts/pruned_hubert_base.pth##g')

    . ./tools/activate_python.sh

    export PYTHONPATH=$PWD
    FAIRDIR=$PWD/tools/fairseq
    export PYTHONPATH=$PYTHONPATH:$FAIRDIR

    python speech_recognition/new/infer.py --config-dir speech_recognition/new/conf \
    --config-name infer task=audio_finetuning task.data=$data_dir common.user_dir=$FAIRDIR/examples/wav2vec \
    task.labels=ltr decoding.type=viterbi \
    decoding.wordscore=${wordscore} decoding.silweight=${silscore} \
    decoding.unique_wer_file=True \
    dataset.gen_subset=$subset dataset.max_tokens=2500000 \
    common_eval.path=$PWD/$model_name common_eval.results_path=$result_path decoding.beam=1500 distributed_training.distributed_world_size=${num_gpus}
done
