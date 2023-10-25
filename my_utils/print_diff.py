import torch
import numpy as np

import json
import pathlib
from argparse import ArgumentParser

from wav2vec2.model import wav2vec2_model

def load_model(ckpt_path, strict=True):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = wav2vec2_model(**ckpt["config"])
    return ckpt["state_dict"]

ckpt_path = "pretrained/wav2vec2_asr-base-ls100.hf.pth"
org = load_model(ckpt_path, strict=False)

ckpt_path = "exp/wav2vec2-base_train_sp0.20_spup5000_lr0.0002_up15000_max50000_layer2layer0.4,8,12_reglr0.02_conv,head,interm_ctc0.0005_mask0.65_chanmask0.5/lightning_logs/version_0/checkpoints/pruned_hubert_base.pth"
pruned = load_model(ckpt_path)

omw = org
smw = pruned

omw_keys = omw.keys()
for omw_key in omw_keys:
    if omw_key not in smw:
        print(omw_key, "not in spinned model")
smw_keys = smw.keys()
for smw_key in smw_keys:
    if smw_key not in omw:
        print(smw_key, "not in original model")

def cosine_similarity_matrix(matrix1, matrix2):
    # Calculate cosine similarity between matrix1 and matrix2
    if matrix1.dim() > 1:
        similarity = torch.nn.functional.cosine_similarity(matrix1, matrix2, dim=1)
    else:
        similarity = torch.nn.functional.cosine_similarity(matrix1, matrix2, dim=0)
    return similarity

def average_cosine_similarity(matrix1, matrix2):
    similarity_matrix = cosine_similarity_matrix(matrix1, matrix2)
    avg_similarity = similarity_matrix.mean()
    return avg_similarity

def average_l2dist(matrix1, matrix2):
    # similarity_matrix = torch.cdist(matrix1, matrix2, p=2.0)
    similarity_matrix = np.power((matrix1 - matrix2), 2)
    avg_similarity = similarity_matrix.mean()
    if avg_similarity < 1e-4:
        avg_similarity = torch.tensor(0)
    return avg_similarity

for omw_key in omw.keys():
    matrix1 = omw[omw_key]
    matrix2 = smw[omw_key]

    if matrix1.shape != matrix2.shape:
        print(omw_key, matrix1.shape, matrix2.shape)
        continue

    # # Calculate average cosine similarity
    # avg_similarity = average_cosine_similarity(matrix1.cpu(), matrix2.cpu())
    avg_similarity = average_l2dist(matrix1.cpu(), matrix2.cpu())

    # print(omw_key, avg_similarity)