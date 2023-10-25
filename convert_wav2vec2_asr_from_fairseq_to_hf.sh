#!/usr/bin/env bash
hf_name="pretrained/fairseq/converted_to_hf"
ckpt="pretrained/fairseq/ted100h_small.pt"
dict="data/librispeech/dev_clean/dict.ltr.txt"

curPath=$(pwd)
mkdir -p ${curPath}/data/temp
mkdir -p $hf_name

cp ${dict} ${curPath}/data/temp/dict.ltr.txt

# load a config that is equal to the config of the model you wish to convert
python -c "from transformers import Wav2Vec2Config; config = Wav2Vec2Config.from_pretrained('facebook/wav2vec2-base-960h'); config.save_pretrained('./');"

# fine-tuned
eval "python tools/venv/envs/dphubert/lib/python3.10/site-packages/transformers/models/wav2vec2/convert_wav2vec2_original_pytorch_checkpoint_to_pytorch.py --pytorch_dump_folder ${hf_name} --checkpoint_path ${ckpt} --config_path ./config.json --dict_path ${curPath}/data/temp/dict.ltr.txt"

mv $hf_name/pytorch_model.bin "pretrained/wav2vec2_asr-base-ted100.hf.pth"