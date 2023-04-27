from email.policy import default
import logging
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP, TransformerEncoder
from fairseq.modules import LayerNorm
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import safe_getattr, safe_hasattr
from fairseq.nat_encoder_generator import DataOut
from fairseq.models.nat.nonautoregressive_pretrain_model import (
    NATPretrainedModel,
    base_architecture,
)


              
import os
from fairseq.modules import (
    PositionalEmbedding,
)

from fairseq.utils import safe_getattr, safe_hasattr
logger = logging.getLogger(__name__)

from transformers import AutoModel, AutoModelForMaskedLM, AutoConfig
from torch.distributions import Categorical
import numpy as np
import random
from scipy.optimize import linear_sum_assignment as lsa
from fairseq.utils import new_arange

from ctcdecode import CTCBeamDecoder


@register_model("ctcpmlm_rate_selection") 
class CTCPMLMRateSelection(NATPretrainedModel):
    def __init__(self, args, translator, src_dict, tgt_dict):
        super().__init__(args, translator, src_dict, tgt_dict)
        
        
    @staticmethod
    def add_args(parser):
        NATPretrainedModel.add_args(parser)
        # """Add model-specific arguments to the parser."""
        # parser.add_argument(
        #     "--rate-list",
        #     type=float,
        #     nargs="+",
        #     default=[2, 3, 4],
        #     help="the lm loss layer , default is -1 (-1 means last layer)",
        # )         
        
        
    def forward(
        self, src_tokens, src_lengths, tgt_tokens, alignments, update_num,
        pretrained_lm=None, lm_loss_layer=-1, upsampling_rate=2.0, **kwargs
    ):  
        if self.lm_loss and update_num > self.lm_start_step :
            self.do_lm_loss = True
        else:
            self.do_lm_loss = False
            
        if self.lmk_loss and True :  # "True" can replace to another condition
            self.do_lmk_loss = True
        else:
            self.do_lmk_loss = False
        
        if self.source_random_mask :
            src_tokens = self._random_mask(src_tokens, self.random_mask_rate)
    
        
        
        if self.voc_choosen == 2 and self.output_projection_warmup > update_num :
            with torch.no_grad(): 
                logits, output_hidden_states, rate, src_upsample_tokens = self.translation(src_tokens, src_lengths, rate=upsampling_rate, **kwargs)
        else:
            logits, output_hidden_states, rate, src_upsample_tokens = self.translation(src_tokens, src_lengths, rate=upsampling_rate, **kwargs)
        
        if self.voc_choosen == 2:
            logits = self.output_projection_layer(output_hidden_states)
        
        # src_upsample_tokens, rate = self.upsampling(src_tokens, rate)
        
        lprobs = self.get_normalized_probs(
            [logits], log_probs=True
        )
        result = {
            "word_ins": {
                "out": lprobs,
                "tgt": tgt_tokens,
                "mask": None if src_upsample_tokens is None else src_upsample_tokens.ne(self.pad),
                "num_upsampling_rate": rate,
                "ls": self.label_smoothing,
                "nll_loss": True,
                "loss_type": "CTC",
            }
        }
        
        if self.target_random_mask :
            tgt_tokens = self._random_mask(tgt_tokens, self.random_mask_rate) 
                   
        if pretrained_lm is not None :
            lm_loss_output_total=torch.zeros([1], device=logits.device)
            lmk_loss_output_total=torch.zeros([1], device=logits.device)
            lm_iter_num=self.lm_iter_num
            lm_loss_iter=0
            lmk_loss_iter=0
            watch_once_lmloss =  self.watch_lm_loss  
            tgt_hidden_states_prev=None
            tgt_logits_prev=None
            for iter_i in range(lm_iter_num) : 
                if iter_i > 0 :
                    watch_once_lmloss = False  # do not watch lm_iter_num times to reduce training time
                    
                if watch_once_lmloss or self.do_lm_loss or self.do_lmk_loss :            
                    with torch.no_grad():   
                        if self.lm_random_mask :
                            tgt_tokens_mask = self._random_mask(tgt_tokens, self.lm_mask_rate) 
                        else :
                            tgt_tokens_mask = tgt_tokens
                        lm_token_embeddings, lm_bos_embeddings, lm_token_logits = \
                                        self.get_pretrained_embedding(tgt_tokens_mask, pretrained_lm, lm_loss_layer) 
                        if self.lm_loss_type == "DIS" :
                            lm_lprobs = self.get_normalized_probs([lm_token_logits], log_probs=True)
                        else:
                            lm_lprobs = None
                else :
                    break                                   
            
                if watch_once_lmloss or self.do_lm_loss :
                    lm_loss_output = self.compute_lm_loss(output_hidden_states,lprobs,src_upsample_tokens, self.lm_loss_type, \
                                                    tgt_tokens,lm_token_embeddings, lm_lprobs, self.do_lm_loss)
                    lm_loss_output_total += lm_loss_output
                    lm_loss_iter = iter_i
                if watch_once_lmloss or self.do_lmk_loss :
                    lmk_loss_output, tgt_hidden_states_prev, tgt_logits_prev = self.compute_lmk_loss(tgt_tokens, lm_token_embeddings,\
                                                    self.lm_loss_type,lm_lprobs, self.do_lmk_loss, tgt_hidden_states_prev, tgt_logits_prev)
                    lmk_loss_output_total+=lmk_loss_output
                    lmk_loss_iter = iter_i
                                         
            
            result.update(
                {
                    "lm": {
                        "loss": lm_loss_output_total/(lm_loss_iter + 1),
                        "factor": 1.0,
                        "loss_type": "LOSS",
                    }  
                }                            
            )
            result.update(
                {
                    "lmk": {
                        "loss": lmk_loss_output_total/(lmk_loss_iter + 1),
                        "factor": self.lmk_loss_factor,
                        "loss_type": "LOSS",
                    }  
                }                            
            )              
        
        return result


@register_model_architecture(
    "ctcpmlm_rate_selection", "ctcpmlm_rate_selection"
)   
def ctcpmlm_rate_selection(args):
    # args.rate_list = safe_getattr(args, "rate_list", [2, 3, 4])
    base_architecture(args)          