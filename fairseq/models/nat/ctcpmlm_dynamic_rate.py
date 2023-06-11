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

# from ctcdecode import CTCBeamDecoder


@register_model("ctcpmlm_rate_selection") 
class CTCPMLMRateSelection(NATPretrainedModel):
    def __init__(self, args, translator, src_dict, tgt_dict):
        super().__init__(args, translator, src_dict, tgt_dict)
        self.rate_list = args.rate_list
        self.tgt_pad = tgt_dict.pad_index
        
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
            "ctc": {
                "out": lprobs,
                "tgt": tgt_tokens,
                "mask": None if src_upsample_tokens is None else src_upsample_tokens.ne(self.pad),
                "num_upsampling_rate": rate,
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
                                         
            if watch_once_lmloss or self.do_lmk_loss :
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
    
    def forward_inference(self, src_tokens, tgt_tokens,src_lengths, alignments=None, update_num=None, **kwargs):
        
        if self.visualization :
            self.visualize(src_tokens,tgt_tokens,src_lengths)
        
        if self.ctc_beam_decoding:
            raise ValueError("delete ctc_beam_decoding function in forward_inference.  \
                            We can add it from nonautoregressive_pretrain_model, but need to modify.")
            print("=============error============")
        else:
            
            if self.debug :
                upsampling_rate=self.debug_value
                logits, output_hidden_states, rate, src_upsample_tokens= self.translation(src_tokens, src_lengths, rate=upsampling_rate, **kwargs) 
                _scores, _tokens = F.log_softmax(logits, dim=-1).max(-1)  #B x T
                if _tokens.size(1) > 0 :
                    unique_x, indices = torch.unique_consecutive(_tokens, return_inverse=True)
                    indices -= indices.min(dim=1, keepdims=True)[0]
                    remove_duplicate_tokens = torch.full_like(_tokens,self.pad)
                    # remove_duplicate_score = torch.full_like(_scores,self.pad)
                    # _scores  = remove_duplicate_score.scatter_(1, indices, _scores)
                    remove_duplicate_tokens = remove_duplicate_tokens.scatter_(1, indices, _tokens)
                else:
                    remove_duplicate_tokens = _tokens      

                return DataOut(
                    output_tokens=remove_duplicate_tokens,
                    output_scores=_scores,
                    attn=None,
                    step=0,
                    max_step=0,
                    history=None,
                )                
            
            
            scores=[] ; tokens=[] ; sentence_scores=[] ; max_length=0
            for upsampling_rate in self.rate_list :
                logits, output_hidden_states, rate, src_upsample_tokens= self.translation(src_tokens, src_lengths, rate=upsampling_rate, **kwargs) 
                if self.voc_choosen == 2:
                    logits = self.output_projection_layer(output_hidden_states)            

                _scores, _tokens = F.log_softmax(logits, dim=-1).max(-1)  #B x T
                
                _sentence_scores = _scores.mean(dim=1)  # B
                if _tokens.size(1) > max_length :
                    max_length = _tokens.size(1)
                scores += [_scores]  #[[BxT],[BxT],...[BxT]]
                tokens += [_tokens]
                sentence_scores +=[_sentence_scores]
            
            for i in range(len(self.rate_list)):
                tokens[i] = torch.nn.functional.pad(tokens[i], (0, max_length - tokens[i].size(1)),
                                                 mode='constant', value=self.tgt_pad)
                scores[i] = torch.nn.functional.pad(scores[i], (0, max_length - scores[i].size(1)), 
                                                 mode='constant', value=0)
                
            if len(tokens) > 0 :
                tokens = torch.stack(tokens)  
                scores = torch.stack(scores) 
                sentence_scores = torch.stack(sentence_scores)                           
            B, L = tokens.size(1),tokens.size(2)
            rate_max_lprob, max_idx = torch.max(sentence_scores, dim = 0)
            _tokens = torch.gather(tokens, 0, max_idx.unsqueeze(0).unsqueeze(-1).expand(1,B,L)).squeeze(0)
            _scores = torch.gather(scores, 0, max_idx.unsqueeze(0).unsqueeze(-1).expand(1,B,L)).squeeze(0)     
        
        
        if _tokens.size(1) > 0 :
            unique_x, indices = torch.unique_consecutive(_tokens, return_inverse=True)
            indices -= indices.min(dim=1, keepdims=True)[0]
            remove_duplicate_tokens = torch.full_like(_tokens,self.pad)
            # remove_duplicate_score = torch.full_like(_scores,self.pad)
            # _scores  = remove_duplicate_score.scatter_(1, indices, _scores)
            remove_duplicate_tokens = remove_duplicate_tokens.scatter_(1, indices, _tokens)
        else:
            remove_duplicate_tokens = _tokens      

        return DataOut(
            output_tokens=remove_duplicate_tokens,
            output_scores=_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )              
    


@register_model_architecture(
    "ctcpmlm_rate_selection", "ctcpmlm_rate_selection"
)   
def ctcpmlm_rate_selection(args):
    base_architecture(args)          

 

class RatePredictor(nn.Module):
    def __init__(self, config, encoder_embed_dim , num_rate=3, bos_idx=None, pad_idx=None):
        super().__init__()
        from transformers import BertConfig, BertModel
        self.rate_bert = BertModel(config)
        self.rate_classifier = nn.Linear(encoder_embed_dim, num_rate)
        self.bos_idx = bos_idx
        self.pad_idx = pad_idx
    def forward(self, input_tokens):
        # bos = self.bos_idx * torch.ones(input_tokens.shape[0], 1, dtype=torch.long, device=input_tokens.device)
        # input_tokens = torch.cat((bos, input_tokens), dim=1) 
        attention_mask = input_tokens.ne(self.pad_idx)
        output_rate_bert = self.rate_bert(input_ids = input_tokens, 
                                attention_mask=attention_mask,  #encoder_attention_mask=attention_mask,
                                output_hidden_states=True, return_dict=True)
        output_rate_classifier = self.rate_classifier(output_rate_bert["pooler_output"])
        
        return output_rate_classifier

            
 
 
@register_model("ctcpmlm_rate_predictor") 
class CTCPMLMRatePredictor(CTCPMLMRateSelection):
    def __init__(self, args, translator, src_dict, tgt_dict):
        super().__init__(args, translator, src_dict, tgt_dict)
        
        
        from transformers import BertConfig, BertModel
        config = BertConfig(
            vocab_size=src_dict.__len__(),
            hidden_size=args.encoder_embed_dim,
            num_hidden_layers=3,
            num_attention_heads=args.encoder_attention_heads,
            intermediate_size=args.encoder_ffn_embed_dim,
        )     
        # self.rate_predictor = BertModel(config)
        self.rate_predictor = RatePredictor(config, args.encoder_embed_dim, args.rate_predictor_classnum, 
                                            self.src_dict.bos_index, self.src_dict.pad_index)
        # Load the pre-trained word embedding weights from the translator model
        for trans_params, rate_params in zip(self.translator.bert.embeddings.parameters(), 
                                             self.rate_predictor.rate_bert.embeddings.parameters()) :
            rate_params = trans_params.detach()  
            
            
        for p in self.translator.parameters():
            p.param_group = "translator"
        for p in self.rate_predictor.parameters():
            p.param_group = "rate_predictor"  
            
        
            
              
        
    
    @staticmethod
    def add_args(parser):
        CTCPMLMRateSelection.add_args(parser)
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--rate-predictor-classnum",
            type=int,
            default=3,
            help="class number of rate predictor, normally is length of rate-list. default is 3 ",
        )                     
        
    
    def ctc_forward(
        self, src_tokens, src_lengths, tgt_tokens, alignments, update_num,
        pretrained_lm=None, lm_loss_layer=-1, upsampling_rate=2.0, **kwargs
    ):  
        
        return super().forward(src_tokens, src_lengths, tgt_tokens, alignments, update_num,
        pretrained_lm, lm_loss_layer, upsampling_rate, **kwargs
    )
    
    def rate_pred_forward(
        self, src_tokens, tgt_rates, **kwargs
    ):        
        output_rate_predictor = self.rate_predictor(src_tokens)
        rate_pred_lprobs= self.get_normalized_probs([output_rate_predictor], log_probs=True)
        
        result = {
            "rate": {
                "out": rate_pred_lprobs,
                "tgt": tgt_rates,
                "nll_loss": True,
                "loss_type": "CE",
            }
        }      
        return result
    def rate_predictor_step(self, num_updates):
        return num_updates % 2 == 1

    def get_groups_for_update(self, num_updates):
        return "rate_predictor" if self.rate_predictor_step(num_updates) else "translator"    
        
          
    def forward_inference(self, src_tokens, tgt_tokens,src_lengths, alignments=None, update_num=None, **kwargs):
        
        if self.visualization :
            self.visualize(src_tokens,tgt_tokens,src_lengths)
        
        if self.ctc_beam_decoding:
            raise ValueError("delete ctc_beam_decoding function in forward_inference.  \
                            We can add it from nonautoregressive_pretrain_model, but need to modify.")
            print("=============error============")
        else:
            
            
            # if self.debug :
            #     upsampling_rate = 2
            #     logits, output_hidden_states, rate, src_upsample_tokens= self.translation(src_tokens, src_lengths, rate=upsampling_rate, **kwargs) 
            #     if self.voc_choosen == 2:
            #         logits = self.output_projection_layer(output_hidden_states)            

            #     _scores, _tokens = F.log_softmax(logits, dim=-1).max(-1)  #B x T
            #     if _tokens.size(1) > 0 :
            #         unique_x, indices = torch.unique_consecutive(_tokens, return_inverse=True)
            #         indices -= indices.min(dim=1, keepdims=True)[0]
            #         remove_duplicate_tokens = torch.full_like(_tokens,self.pad)
            #         remove_duplicate_tokens = remove_duplicate_tokens.scatter_(1, indices, _tokens)
            #     else:
            #         remove_duplicate_tokens = _tokens      

            #     return DataOut(
            #         output_tokens=remove_duplicate_tokens,
            #         output_scores=_scores,
            #         attn=None,
            #         step=0,
            #         max_step=0,
            #         history=None,
            #     )                  
    
                
                
            scores=[] ; tokens=[] ; sentence_scores=[] ; max_length=0
            for upsampling_rate in self.rate_list :
                logits, output_hidden_states, rate, src_upsample_tokens= self.translation(src_tokens, src_lengths, rate=upsampling_rate, **kwargs) 
                if self.voc_choosen == 2:
                    logits = self.output_projection_layer(output_hidden_states)            

                _scores, _tokens = F.log_softmax(logits, dim=-1).max(-1)  #B x T
                
                # _sentence_scores = _scores.mean(dim=1).view(-1,1)  # B x 1
                # sentence_scores = torch.stack((sentence_scores,_sentence_scores))
                if _tokens.size(1) > max_length :
                    max_length = _tokens.size(1)
                scores += [_scores]  #[[BxT],[BxT],...[BxT]]
                tokens += [_tokens]
            
            # if len(sentence_scores) > 0 :
            #     sentence_scores = torch.stack(sentence_scores)   
            for i in range(len(self.rate_list)):
                tokens[i] = torch.nn.functional.pad(tokens[i], (0, max_length - tokens[i].size(1)),
                                                 mode='constant', value=self.tgt_pad)
                scores[i] = torch.nn.functional.pad(scores[i], (0, max_length - scores[i].size(1)), 
                                                 mode='constant', value=0)
                
            if len(tokens) > 0 :
                tokens = torch.stack(tokens)  
                scores = torch.stack(scores)                           
            B, L = tokens.size(1),tokens.size(2)
            rate_outputs = self.rate_pred_forward(src_tokens, None)
            rate_max_lprob, max_idx = torch.max(rate_outputs['rate']['out'], dim = 1)
            _tokens = torch.gather(tokens, 0, max_idx.unsqueeze(0).unsqueeze(-1).expand(1,B,L)).squeeze(0)
            _scores = torch.gather(scores, 0, max_idx.unsqueeze(0).unsqueeze(-1).expand(1,B,L)).squeeze(0)
                
            # logits, output_hidden_states, rate, src_upsample_tokens= self.translation(src_tokens, src_lengths, rate=self.num_upsampling_rate, **kwargs) 
            # if self.voc_choosen == 2:
            #     logits = self.output_projection_layer(output_hidden_states)            

            # _scores, _tokens = F.log_softmax(logits, dim=-1).max(-1)         
        
        
        if _tokens.size(1) > 0 :
            unique_x, indices = torch.unique_consecutive(_tokens, return_inverse=True)
            indices -= indices.min(dim=1, keepdims=True)[0]
            remove_duplicate_tokens = torch.full_like(_tokens,self.pad)
            # remove_duplicate_score = torch.full_like(_scores,self.pad)
            # _scores  = remove_duplicate_score.scatter_(1, indices, _scores)
            remove_duplicate_tokens = remove_duplicate_tokens.scatter_(1, indices, _tokens)
        else:
            remove_duplicate_tokens = _tokens      

        return DataOut(
            output_tokens=remove_duplicate_tokens,
            output_scores=_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )          


@register_model_architecture(
    "ctcpmlm_rate_predictor", "ctcpmlm_rate_predictor"
)   
def ctcpmlm_rate_predictor(args):
    args.rate_predictor_classnum = safe_getattr(args, "rate_predictor_classnum", 3)
    
    base_architecture(args)           