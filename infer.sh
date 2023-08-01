#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

. tools/activate_python.sh

set -e
set -u
set -o pipefail

for subset in "dev_clean"; do
    data_dir=$PWD/data/librispeech/$subset/
    # finetuned_model=$PWD/exp/infer_model.pth
    finetuned_model=$PWD/pretrained/wav2vec2_asr-base-ls100.hf.pth
    inference_result=$PWD/inference_result/
    wordscore=-1
    lmweight=2
    silscore=0
    num_gpus=1

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
    common_eval.path=$finetuned_model decoding.beam=1500 distributed_training.distributed_world_size=${num_gpus}
done