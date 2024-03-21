#!/bin/bash

# first source conda.sh, and then
# activate your conda environment

. tools/activate_python.sh

# shared config
tsv_dir=data/librispeech/train_960        # data path
train_subset=train          # train subset name: train960, train100
teacher_ckpt=pretrained/wav2vec2_asr-large-ls960.hf.pth    # checkpoint path
student_ckpt=${teacher_ckpt}    # student initialization, same as teacher
distill_layers=0,8,16,24         # use period to separate groups where each group shares the same linear layer: [0], [4, 8, 12]
distill_mode=layer2layer        # "layer2layer", "predlayer"
l2_weight=0             # weight for L2 loss
l1_weight=1             # weight for L1 loss
cos_weight=1            # weight for cosine similarity
cos_type=raw            # "raw", "log_sig"

# loss weight config
ctc_weight=0.0005
distill_weight=1.0      # distill loss weight

# masking config
mask_prob=0.2
mask_channel_prob=0.2

# distill config
lr=0.0002               # learning rate
warmup=15000            # warmup steps
max=50000               # max update steps
pruning_units=conv,head,interm      # conv,head,interm,attlayer,ffnlayer
reg_lr=0.02             # learning rate for regularization params
sparsity_warmup=5000    # warmup steps for sparsity; sparsity will linearly increase from 0 to target
threshold=0.8           # threshold for pruning
target_sparsity=0.70    # final target sparsity

# parameters regularization config
param_reg_type="l2"

# exp directory
root_dir=exp/w2v2large_960to960_dp_reg${param_reg_type}_sp${target_sparsity}_th${threshold}

if [ -d "$root_dir" ]; then
  echo "Directory exists. Deleting $root_dir"
  rm -rf "$root_dir"
else
  echo "$root_dir does not exist."
fi

# wandb project
project_name="dphubert-param-reg"

dict_type="fairseq"

if [ "$dict_type" == "hf" ]; then
  # Move the file if dict_type is "hf"
  cp "data/hf_dict.txt" "$tsv_dir/dict.ltr.txt"
elif [ "$dict_type" == "fairseq" ]; then
  cp "data/fairseq_dict.txt" "$tsv_dir/dict.ltr.txt"
fi

# Training step 1: distill
mkdir -p ${root_dir}

python distill.py \
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
    --gpus 4 \
    --accum_grad 2 \
    --precision 16 \
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
    --ctc_weight ${ctc_weight} \
    --distill_weight ${distill_weight} \
    --mask_prob ${mask_prob} \
    --mask_channel_prob ${mask_channel_prob} \
    --param_reg_type ${param_reg_type} \
    --project_name ${project_name} \
    --threshold ${threshold} \
    --sparsity_warmup_updates ${sparsity_warmup} 2>&1 | tee ${root_dir}/distill.log || exit 1;

# prune and save model
python prune.py \
    --distilled_ckpt ${root_dir}/ckpts/*.ckpt \
    --original_ckpt ${student_ckpt} || exit 1;

. ./infer.sh --model_name $root_dir/ckpts/pruned_hubert_base.pth
. ./infer_ted.sh --model_name $root_dir/ckpts/pruned_hubert_base.pth
. ./infer_l2arc.sh --model_name $root_dir/ckpts/pruned_hubert_base.pth
. ./infer_cv.sh --model_name $root_dir/ckpts/pruned_hubert_base.pth
. ./infer_chime.sh --model_name $root_dir/ckpts/pruned_hubert_base.pth
