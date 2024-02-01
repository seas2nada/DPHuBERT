# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools as it
from typing import Any, Dict, List

import torch
from fairseq.data.dictionary import Dictionary
from fairseq.models.fairseq_model import FairseqModel


class BaseDecoder:
    def __init__(self, tgt_dict: Dictionary) -> None:
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)

        self.blank = (
            tgt_dict.index("<ctc_blank>")
            if "<ctc_blank>" in tgt_dict.indices
            else tgt_dict.bos()
        )
        if "<sep>" in tgt_dict.indices:
            self.silence = tgt_dict.index("<sep>")
        elif "|" in tgt_dict.indices:
            self.silence = tgt_dict.index("|")
        else:
            self.silence = tgt_dict.eos()

    def generate(
        self, models: List[FairseqModel], sample: Dict[str, Any], **unused
    ) -> List[List[Dict[str, torch.LongTensor]]]:
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        emissions = self.get_emissions(models, encoder_input)
        
        emm = emissions[0].argmax(-1)
        emm = emm.unique_consecutive()
        emm = emm[emm != 28]
        with open('/home/Workspace/DPHuBERT/data/hf_dict.txt', 'r') as f:
            dicts = f.readlines()
        # convert_dict = {0 : '<bos>', 1 : '<eos>'}
        convert_dict = {}
        for pair in dicts:
            k, v = pair.strip().split(' ')
            convert_dict[int(v)] = k
            # if k != "|":
            #     convert_dict[int(v) + 1] = k
            # elif k == "|":
            #     convert_dict[28] = "|"
        letters=[]
        for e in emm:
            letters.append(convert_dict[int(e)])
        print("".join(letters))
        exit()
        return self.decode(emissions)

    def get_emissions(
        self,
        models: List[FairseqModel],
        encoder_input: Dict[str, Any],
    ) -> torch.FloatTensor:
        model = models[0]

        waveforms = encoder_input['source']
        lengths =  torch.zeros(len(encoder_input['padding_mask'])).to(torch.int).to(waveforms.device)
        for i, pm in enumerate(encoder_input['padding_mask']):
            lengths[i] = len(pm) - pm.sum()
        
        encoder_out, _ = model(waveforms, lengths)
        if hasattr(model, "get_logits"):
            emissions = model.get_logits(encoder_out)
        else:
            emissions = model.get_normalized_probs(encoder_out, log_probs=True)
        return emissions.float().cpu().contiguous()

    def get_tokens(self, idxs: torch.IntTensor) -> torch.LongTensor:
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)
        return torch.LongTensor(list(idxs))

    def decode(
        self,
        emissions: torch.FloatTensor,
    ) -> List[List[Dict[str, torch.LongTensor]]]:
        raise NotImplementedError
