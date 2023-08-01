#!/bin/bash

# first source conda.sh, and then
# activate your conda environment

set -x

# shared config
tsv_dir=data/librispeech        # data path
train_subset=train960           # train subset name: train960, train100
teacher_ckpt=pretrained/wav2vec2-large-ls960.fairseq.pth    # checkpoint path
student_ckpt=${teacher_ckpt}    # student initialization, same as teacher
distill_layers=0.4,8,12,16,20,24         # use period to separate groups where each group shares the same linear layer: [0], [4, 8, 12]
distill_mode=layer2layer        # "layer2layer", "predlayer"
l2_weight=0             # weight for L2 loss
l1_weight=1             # weight for L1 loss
cos_weight=1            # weight for cosine similarity
cos_type=raw            # "raw", "log_sig"

# distill config
lr=0.0002               # learning rate
warmup=50000            # warmup steps
max=140000               # max update steps
pruning_units=conv,head,interm      # conv,head,interm,attlayer,ffnlayer
reg_lr=0.02             # learning rate for regularization params
target_sparsity=0.75    # final target sparsity
sparsity_warmup=5000    # warmup steps for sparsity; sparsity will linearly increase from 0 to target
root_dir=exp/wav2vec2-large_${train_subset}_sp${target_sparsity}_spup${sparsity_warmup}_lr${lr}_up${warmup}_max${max}_${distill_mode}${distill_layers}_reglr${reg_lr}_${pruning_units}

# final distill config
final_lr=0.0001         # learning rate for final distillation (training step 2)
final_warmup=14000       # warmup steps
final_max=70000         # max update steps
final_exp_dir=${root_dir}/lr${final_lr}_up${final_warmup}_max${final_max}


# Training step 1: distill
mkdir -p ${root_dir}

python distill.py \
    --tsv_dir ${tsv_dir} \
    --train_subset ${train_subset} \
    --seconds_per_batch 60 \
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
    --accum_grad 3 \
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
    --sparsity_warmup_updates ${sparsity_warmup} 2>&1 | tee ${root_dir}/distill.log || exit 1;

# prune and save model
python prune.py \
    --distilled_ckpt ${root_dir}/ckpts/*.ckpt \
    --original_ckpt ${student_ckpt} || exit 1;

mv ${root_dir}/ckpts/pruned_hubert_base.pth ${root_dir}/ckpts/pruned_wav2vec2_large.pth


# Training step 2: final distill
pruned_ckpt=${root_dir}/ckpts/pruned_wav2vec2_large.pth
mkdir -p ${final_exp_dir}

python final_distill.py \
    --tsv_dir ${tsv_dir} \
    --train_subset ${train_subset} \
    --seconds_per_batch 60 \
    --num_workers 12 \
    --exp_dir ${final_exp_dir} \
    --log_interval 50 \
    --learning_rate ${final_lr} \
    --weight_decay 0.0 \
    --warmup_updates ${final_warmup} \
    --max_updates ${final_max} \
    --clip_norm 10.0 \
    --num_nodes 1 \
    --gpus 4 \
    --accum_grad 3 \
    --precision 16 \
    --teacher_ckpt ${teacher_ckpt} \
    --student_ckpt ${pruned_ckpt} \
    --distill_layers ${distill_layers} \
    --distill_mode ${distill_mode} \
    --l2_weight ${l2_weight} \
    --l1_weight ${l1_weight} \
    --cos_weight ${cos_weight} \
    --cos_type ${cos_type} 2>&1 | tee ${final_exp_dir}/final_distill.log || exit 1;

# save final model and config
python save_final_ckpt.py \
    --config_path ${pruned_ckpt} \
    --ckpt_after_final_distill ${final_exp_dir}/ckpts/*.ckpt || exit 1;
