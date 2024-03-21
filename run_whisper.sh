#!/bin/bash

# first source conda.sh, and then
# activate your conda environment

. tools/activate_python.sh

# shared config
tsv_dir=data/librispeech/train_clean_100        # data path
train_subset=train          # train subset name: train960, train100
teacher_ckpt=pretrained/whisper-medium.hf.pth    # checkpoint path
student_ckpt=${teacher_ckpt}    # student initialization, same as teacher
distill_layers=0,8,16,24         # use period to separate groups where each group shares the same linear layer: [0], [4, 8, 12]
distill_mode=layer2layer        # "layer2layer", "predlayer"
l2_weight=0             # weight for L2 loss
l1_weight=1             # weight for L1 loss
cos_weight=1            # weight for cosine similarity
cos_type=raw            # "raw", "log_sig"

# loss weight config
ce_weight=0.5
distill_weight=1.0      # distill loss weight

# masking config
mask_prob=0.2
mask_channel_prob=0.2

# distill config
lr=0.0002              # learning rate
warmup=10000            # warmup steps
max=30000               # max update steps
pruning_units=conv,head,interm      # conv,head,interm,attlayer,ffnlayer
reg_lr=0.02             # learning rate for regularization params
target_sparsity=0.20    # final target sparsity
sparsity_warmup=3000    # warmup steps for sparsity; sparsity will linearly increase from 0 to target
threshold=0.0          # threshold for pruning

# # parameters regularization config
param_reg_type="none"

language="es+fr+pt+de"

# exp directory
root_dir=exp/whisper_medium_${language}_reg${param_reg_type}_8gpu_max${max}_sp${target_sparsity}
# root_dir=exp/whisper_test

if [ -d "$root_dir" ]; then
  echo "Directory exists. Deleting $root_dir"
  rm -rf "$root_dir"
else
  echo "$root_dir does not exist."
fi

# wandb project
project_name="dphubert-param-reg"

# Training step 1: distill
mkdir -p ${root_dir}

export TOKENIZERS_PARALLELISM=false
python whisper_distill.py \
    --tsv_dir ${tsv_dir} \
    --label_dir ${tsv_dir} \
    --train_subset ${train_subset} \
    --seconds_per_batch 160 \
    --num_workers 12 \
    --exp_dir ${root_dir} \
    --log_interval 50 \
    --learning_rate ${lr} \
    --weight_decay 0.0 \
    --warmup_updates ${warmup} \
    --max_updates ${max} \
    --clip_norm 10.0 \
    --num_nodes 1 \
    --gpus 8 \
    --accum_grad 1 \
    --teacher_ckpt ${teacher_ckpt} \
    --student_ckpt ${student_ckpt} \
    --distill_layers ${distill_layers} \
    --distill_mode ${distill_mode} \
    --l2_weight ${l2_weight} \
    --l1_weight ${l1_weight} \
    --cos_weight ${cos_weight} \
    --cos_type ${cos_type} \
    --pruning_units ${pruning_units} \
    --reg_learning_rate ${reg_lr} \
    --target_sparsity ${target_sparsity} \
    --ce_weight ${ce_weight} \
    --distill_weight ${distill_weight} \
    --mask_prob ${mask_prob} \
    --mask_channel_prob ${mask_channel_prob} \
    --param_reg_type ${param_reg_type} \
    --project_name ${project_name} \
    --language $language \
    --whisper_model_name "openai/whisper-medium" \
    --batch_size 32 \
    --threshold ${threshold} \
    --sparsity_warmup_updates ${sparsity_warmup} 2>&1 | tee ${root_dir}/distill.log || exit 1;

# prune and save model
python prune_whisper.py \
    --distilled_ckpt ${root_dir}/ckpts/*.ckpt \
    --original_ckpt ${student_ckpt} || exit 1;

for language in "en_us" "es_419" "fr_fr" "pt_br" "de_de" "ko_kr" "it_it"; do
  . ./infer_whisper_pruned.sh --exp_dir ${root_dir} --model_name ${root_dir}/ckpts/pruned_hubert_base.pth --language $language --whisper_model_name "openai/whisper-medium" --eval_metric "wer"
done
for language in "ja_jp" "cmn_hans_cn"; do
  . ./infer_whisper_pruned.sh --exp_dir ${root_dir} --model_name ${root_dir}/ckpts/pruned_hubert_base.pth --language $language --whisper_model_name "openai/whisper-medium" --eval_metric "cer"
done

for language in "es" "fr" "pt" "de" "ko" "en" "it"; do 
bash infer_whisper.sh --exp_dir ${root_dir} --model_name ${root_dir}/ckpts/pruned_hubert_base.pth --language $language --whisper_model_name "openai/whisper-medium" --eval_metric "wer" --dataset "mozilla-foundation/common_voice_16_1"; done

for language in "ja" "zh-CN"; do 
bash infer_whisper.sh --exp_dir ${root_dir} --model_name ${root_dir}/ckpts/pruned_hubert_base.pth --language $language --whisper_model_name "openai/whisper-medium" --eval_metric "cer" --dataset "mozilla-foundation/common_voice_16_1"; done