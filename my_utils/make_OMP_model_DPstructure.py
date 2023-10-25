import torch

from wav2vec2.model import (
    wav2vec2_model,
)

out_name = "trained_models/omp_pruned_w2v2_base_ls100.pth"

model = torch.load("trained_models/pruned_hubert_base.pth")
config = model['config']
model = wav2vec2_model(**config)

import copy
before_model_dict = copy.deepcopy(model.state_dict())

model_org = torch.load("pretrained/wav2vec2_asr-base-ls100.hf.pth")
model_org_dict = model_org['state_dict']

not_pruned = []
for n, p in model.named_parameters():
    if n == "feature_extractor.dummy_weight":
        not_pruned.append(n)
        continue

    org_p = model_org_dict[n]
    
    if p.shape != model_org_dict[n].shape:
        l2_magnitude = torch.sqrt(org_p ** 2)

        if p.dim() == 1:
            num_pruned = len(org_p) - len(p)

            # Getting indices of smallest L2 maginitudes
            smallest_indices = torch.argsort(l2_magnitude)[:num_pruned]

            # Creating a mask to remove the selected indices
            mask = torch.ones(len(org_p), dtype=bool)
            mask[smallest_indices] = False
            updated_p = org_p[mask]

            with torch.no_grad():
                p.copy_(updated_p.data)

        elif p.dim() == 2:
            if p.shape[0] != org_p.shape[0]:
                num_pruned = org_p.shape[0] - p.shape[0]
                
                smallest_indices = torch.ones(org_p.shape[1], num_pruned).long()
                
                l2_magnitude = l2_magnitude.transpose(0, 1)
                for i, mag in enumerate(l2_magnitude):
                    smallest_indices[i] = torch.argsort(mag)[:num_pruned]
                smallest_indices = smallest_indices.transpose(0, 1)
                l2_magnitude = l2_magnitude.transpose(0, 1)

                mask = torch.ones(org_p.size(), dtype=bool)
                for r_ind in range(org_p.shape[1]):
                    mask[smallest_indices[:, r_ind], r_ind] = False

                updated_p = torch.zeros((org_p.shape[0] - num_pruned, org_p.shape[1]), dtype=org_p.dtype)
                for i in range(org_p.shape[1]):
                    updated_p[:, i] = org_p[mask[:, i], i]

            elif p.shape[1] != org_p.shape[1]:
                num_pruned = org_p.shape[1] - p.shape[1]
                
                smallest_indices = torch.ones(org_p.shape[0], num_pruned).long()
                
                for i, mag in enumerate(l2_magnitude):
                    smallest_indices[i] = torch.argsort(mag)[:num_pruned]

                mask = torch.ones(org_p.size(), dtype=bool)
                for l_ind in range(org_p.shape[0]):
                    mask[l_ind, smallest_indices[l_ind, :]] = False

                updated_p = torch.zeros((org_p.shape[0], org_p.shape[1] - num_pruned), dtype=org_p.dtype)
                for i in range(org_p.shape[0]):
                    updated_p[i, :] = org_p[i, mask[i, :]]

            with torch.no_grad():
                p.copy_(updated_p.data)

        elif p.dim() == 3:
            if p.shape[0] != org_p.shape[0]:
                p_ = p.squeeze(1)
                org_p_ = org_p.squeeze(1)
                l2_magnitude = l2_magnitude.squeeze(1)

                num_pruned = org_p_.shape[0] - p_.shape[0]
                
                smallest_indices = torch.ones(org_p_.shape[1], num_pruned).long()
                
                l2_magnitude = l2_magnitude.transpose(0, 1)
                for i, mag in enumerate(l2_magnitude):
                    smallest_indices[i] = torch.argsort(mag)[:num_pruned]
                smallest_indices = smallest_indices.transpose(0, 1)
                l2_magnitude = l2_magnitude.transpose(0, 1)

                mask = torch.ones(org_p_.size(), dtype=bool)
                for r_ind in range(org_p_.shape[1]):
                    mask[smallest_indices[:, r_ind], r_ind] = False

                updated_p = torch.zeros((org_p_.shape[0] - num_pruned, org_p_.shape[1]), dtype=org_p_.dtype)
                for i in range(org_p_.shape[1]):
                    updated_p[:, i] = org_p_[mask[:, i], i]
                updated_p = updated_p.unsqueeze(1)

            elif p.shape[1] != org_p.shape[1]:
                p_rest = p.shape[1]
                org_p_rest = org_p.shape[1]

                p_ = p.transpose(1, 2).contiguous().view(-1, p_rest)
                org_p_ = org_p.transpose(1, 2).contiguous().view(-1, org_p_rest)
                l2_magnitude = l2_magnitude.transpose(1, 2).contiguous().view(-1, org_p_rest)

                num_pruned = org_p_.shape[1] - p_.shape[1]
                
                smallest_indices = torch.ones(org_p_.shape[0], num_pruned).long()
                
                for i, mag in enumerate(l2_magnitude):
                    smallest_indices[i] = torch.argsort(mag)[:num_pruned]

                mask = torch.ones(org_p_.size(), dtype=bool)
                for l_ind in range(org_p_.shape[0]):
                    mask[l_ind, smallest_indices[l_ind, :]] = False

                updated_p = torch.zeros((org_p_.shape[0], org_p_.shape[1] - num_pruned), dtype=org_p_.dtype)
                for i in range(org_p_.shape[0]):
                    updated_p[i, :] = org_p_[i, mask[i, :]]
                
                updated_p = updated_p.view(p.shape[0], p.shape[2], p.shape[1]).transpose(1, 2)
            
            with torch.no_grad():
                p.copy_(updated_p.data)

    else:
        not_pruned.append(n)
        with torch.no_grad():
            p.copy_(org_p.data)

torch.save(
        {
            'state_dict': model.state_dict(),
            'config': config,
        }, 
        out_name
    )
