#!/bin/bash

# first source conda.sh, and then
# activate your conda environment

. tools/activate_python.sh

set -x

# shared config
tsv_dir=data/librispeech/train_clean_100        # data path
train_subset=train          # train subset name: train960, train100
teacher_ckpt=pretrained/wav2vec2_asr-base-ls100.hf.pth    # checkpoint path
student_ckpt=trained_models/omp_pruned_w2v2_base_ls100.pth    # student initialization, same as teacher
distill_layers=0.4,8,12         # use period to separate groups where each group shares the same linear layer: [0], [4, 8, 12]
distill_mode=layer2layer        # "layer2layer", "predlayer"
l2_weight=0             # weight for L2 loss
l1_weight=1             # weight for L1 loss
cos_weight=1            # weight for cosine similarity
cos_type=raw            # "raw", "log_sig"

# loss weight config
ctc_weight=1
distill_weight=0.0      # distill loss weight

# masking config
mask_prob=0.2
mask_channel_prob=0.2

# distill config
lr=0.0002               # learning rate
warmup=15000            # warmup steps
max=80000               # max update steps
pruning_units=conv,head,interm      # conv,head,interm,attlayer,ffnlayer
reg_lr=0.02             # learning rate for regularization params
target_sparsity=0.00    # final target sparsity
root_dir=exp/wav2vec2-base_${train_subset}_sp${target_sparsity}_spup${sparsity_warmup}_lr${lr}_up${warmup}_max${max}_${distill_mode}${distill_layers}_reglr${reg_lr}_${pruning_units}_ctc${ctc_weight}_mask${mask_prob}_chanmask${mask_channel_prob}
rm -rf root_dir

# save_checkpoint
save_checkpoint="last"

# spk2info
spk2info="./data/spk2info.dict"

# Training step 1: distill
mkdir -p ${root_dir}

python train_ctc.py \
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
    --accum_grad 1 \
    --precision 16 \
    --teacher_ckpt ${teacher_ckpt} \
    --student_ckpt ${student_ckpt} \
    --distill_layers ${distill_layers} \
    --distill_mode ${distill_mode} \
    --l2_weight ${l2_weight} \
    --l1_weight ${l1_weight} \
    --cos_weight ${cos_weight} \
    --ctc_weight ${ctc_weight} \
    --distill_weight ${distill_weight} \
    --mask_prob ${mask_prob} \
    --mask_channel_prob ${mask_channel_prob} \
    --cos_type ${cos_type} 2>&1 | tee ${final_exp_dir}/final_distill.log || exit 1;

. ./infer.sh --model_name $root_dir/lightning_logs/version_0/checkpoints/*.ckpt