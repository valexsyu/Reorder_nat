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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt  



# Define the neural network model
class PolyNet(nn.Module):
    def __init__(self,output_dim):
        super(PolyNet, self).__init__()
        self.fc1 = nn.Linear(2, 100 )
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, output_dim)
    def forward(self, x):
        x = torch.column_stack((x, x**2))        
        x = torch.relu(self.fc1(x))
        x = torch.nn.functional.normalize(x)
        x = torch.relu(self.fc2(x))
        x = torch.nn.functional.normalize(x)
        x = self.fc3(x)
        return x




@register_model("ctcpmlm_rate_pred") 
class CTCPMLMRatePred(NATPretrainedModel):
    def __init__(self, args, translator, src_dict, tgt_dict):
        super().__init__(args, translator, src_dict, tgt_dict)
        self.num_rate_level = args.num_rate_level
        self.pseudo_rate_start_step = args.pseudo_rate_start_step
        self.initial_target_rate_level = args.initial_target_rate_level
        if self.initial_target_rate_level > self.num_rate_level :
            print("initial_target_rate_level must be smaller than num_rate_level")
            import pdb;pdb.set_trace()
        self.rate_levels = torch.arange(start=2, end=6, step=self.num_rate_level)
        self.length_rate_predictor = PolyNet(self.num_rate_level)

    @staticmethod
    def add_args(parser):
        super().add_args(parser)
        """Add model-specific arguments to the parser."""
 
        parser.add_argument(
            "--pseudo-rate-start-step",
            type=int,
            default=10000,
            help="st",
        )       
        parser.add_argument(
            "--initial-target-rate-level",
            type=int,
            default=3,
            help="initial target rate level",
        )       
        parser.add_argument(
            "--num-rate-level",
            type=int,
            default=5,
            help="the rate level number",
        )  
    
    
    def generate_pseudo_rate(self, lengths) :
        B= lengths.size(0)
        
        
        
        
    def forward(
        self, src_tokens, src_lengths, tgt_tokens, alignments, update_num,
        pretrained_lm=None, lm_loss_layer=-1, **kwargs
    ):  
        if update_num > self.pseudo_rate_start_step :
            print("generate peudo target rate")
            pseudo_target_rate = generate_pseudo_rate(src_lengths) 
            
        else :
            print("use initial target rate")
            pseudo_target_rate = 2
        
        
        
        data_type = next(self.parameters()).dtype       
        batch_src_length = src_lengths.type(data_type).to(src_lengths.device).unsqueeze(1)
        pred_rate = self.length_rate_predictor(batch_src_length)     
        if self.debug:
            import pdb;pdb.set_trace()
            print("EEE")
            
            # if update_num % 100 == 0:
            #     qq=torch.arange(2,15).type(data_type).to(pred_rate.device)
            #     gg=self.length_rate_predictor(qq) 
            #     qq_np=qq.cpu().detach().numpy()
            #     gg_np=gg.squeeze(-1).cpu().detach().numpy()
            #     with open("qqqqqq.txt","a") as f:
            #         np.savetxt(f,(gg_np,qq_np))        
        self.num_upsampling_rate = pred_rate
        
        result = super().forward( src_tokens, src_lengths, tgt_tokens, alignments, update_num,
                                pretrained_lm, lm_loss_layer)
        return result   

    def ctc_sentence_loss(self, src_tokens, src_lengths, targets, masks=None, num_upsampling_rate=2, factor=1.0,  
    ):              
        previous_rate = num_upsampling_rate
        logits, output_hidden_states, rate, src_upsample_tokens = self.translation(src_tokens, src_lengths, upsampling_flag=True)
        
           
        return result
        
    def forward_inference(self, src_tokens, tgt_tokens,src_lengths, 
                        alignments=None, update_num=None, **kwargs):  
        data_type = next(self.parameters()).dtype       
        batch_src_length = src_lengths.type(data_type).to(src_lengths.device).unsqueeze(1)
        pred_rate = self.length_rate_predictor(batch_src_length)       
        self.num_upsampling_rate = pred_rate
        if self.debug :
            pred_rate_np = pred_rate.cpu().numpy()
            batch_src_length_np = batch_src_length.cpu().numpy()
            np.savetxt('pred_rate.txt', pred_rate_np)
            np.savetxt('length.txt', batch_src_length_np)
        return super().forward_inference(src_tokens, tgt_tokens,src_lengths, 
                        alignments, update_num, **kwargs)

    def upsampling(self, source, rate):     
        def dynamic_upsample_token(x, insert_mask=False , rate=2, insertion_position='uniform'):
            
            B, L = x.size(0), x.size(1)
            if torch.is_tensor(rate):
                new_length = (L * rate).max().int().item()
            else:
                new_length = torch.Tensor([L * rate]).int().item()
            # new_length = int(L * rate)   # 50*2.3=114.9999999
            pad = self.src_dict.pad()
            bos = self.src_dict.bos()
            eos = self.src_dict.eos()      
            ### the mask is True when padding/bos/eos 
            
            if self.has_eos :            
                mask = ~(
                    x.ne(pad) & x.ne(bos) # keep the eos and it will upsample by rate
                )   
            else :
                mask = ~(
                    x.ne(pad) & x.ne(bos) & x.ne(eos) # old version for eos is padding and masked
                )                     
                        
            l = (x.new_ones(B, L) * rate).float()
            l = l.masked_fill(mask, 0)
            e = torch.cumsum(l, 1)
            c = e - l / 2
            t = e[:, -1].ceil().long()
            # pdb.set_trace()

            # t = new_arange(t, t.max())[None, :].expand(l.size(0), -1)  # B x L2
            t = new_arange(t, new_length)[None, :].expand(l.size(0), -1)  # B x L2

            t_mask = t >= e[:, -1:]   # target padding mask
            
            if insertion_position == 'uniform':            
                w = -(t[:, None, :] - c[:, :, None]) ** 2 / 0.3

                w = w.float()
                w = w.masked_fill(mask.unsqueeze(-1), -10000.0)
                w = w.masked_fill(t_mask.unsqueeze(1), -10000.0)
                t_w = F.softmax(w, dim=1)   # B x L x L2
                try:
                    new_location = t_w.argmax(-1)
                except:
                    import pdb;pdb.set_trace()
                    print("EEE")
                
                if insert_mask:
                    
                    new_t_w = F.one_hot(new_location, num_classes=new_length).masked_fill(mask.unsqueeze(-1), 0)
                    
                else:
                    new_t_w = F.one_hot(new_location, num_classes=new_length).masked_fill(mask.unsqueeze(-1), 0)
                    new_location = torch.cat((new_location, torch.ones((B, 1)).to(new_location)*new_length), 1)
                    new_t_w[(torch.arange(0, new_length, dtype=torch.float32).unsqueeze(0).repeat(B, L, 1).to(new_location) >= new_location[:, :-1].unsqueeze(-1)) &
                            (torch.arange(0, new_length, dtype=torch.float32).unsqueeze(0).repeat(B, L, 1).to(new_location) < new_location[:, 1:].unsqueeze(-1))] = 1
                    
                t_x = torch.einsum('bst,bs->bt', new_t_w.to(x).float(), x.float()).long().to(x)
                # t_x = torch.matmul(x.float(), new_t_w.to(x).float()).to(x)
                

                if insert_mask:
                    t_x[torch.where(t_x == pad)] = self.mask
                    
                t_x = t_x.masked_fill(t_mask, pad)
                return t_x, t_mask, w, t_w, new_t_w, new_location                
            
            elif insertion_position == 'left':
                t_x = x.new_zeros(x.size(0), new_length) 
                seq_lengths = L - mask.sum(axis=1)
                mask_length = ((rate-1)*seq_lengths).long()
                new_location = torch.arange(0, L).unsqueeze(0).repeat(B, 1).to(x) + mask_length.unsqueeze(-1) + 1
                new_t_w = F.one_hot(new_location, num_classes=new_length).masked_fill(mask.unsqueeze(-1), 0)
                
                t_x = torch.einsum('bst,bs->bt', new_t_w.to(x).float(), x.float()).to(x)
                
                t_x[torch.where(t_x == pad)] = 4
                
                t_x = t_x.masked_fill(t_mask, pad)
                
                return t_x, t_mask, [], [], new_t_w, new_location       
            elif insertion_position == 'right':
                t_x = torch.full((x.size(0), new_length), self.mask).to(x)
                # t_x = x.new_ones(x.size(0), new_length) * 4
                t_x[:, :L] = x.masked_fill(mask, self.mask)
                
                t_x = t_x.masked_fill(t_mask, pad)
                
                return t_x, t_mask, [], [], [], []                
            else:
                import pdb;pdb.set_trace()
                print("insertion_position is not well define")             
           
        if torch.is_tensor(rate): ##pre_dict_rate
            B, L = source.size(0), source.size(1)
            new_length = (L * rate).max().int().item()
            if new_length > self.translator.config.max_position_embeddings :
                rate= torch.full_like(rate, self.translator.config.max_position_embeddings/L)              
            if self.upsample_fill_mask :
                insert_mask = True
                t_x, t_mask, w, t_w, new_t_w, new_location = dynamic_upsample_token(source, insert_mask , rate, insertion_position=self.insert_position)  
                return t_x, rate                      
            else:    
                                          
                insert_mask = False
                t_x, t_mask, w, t_w, new_t_w, new_location = dynamic_upsample_token(source, insert_mask , rate) 
                return t_x, rate 


           
           
@register_model_architecture(
    "ctcpmlm_rate_pred", "ctcpmlm_rate_pred"
)   
def ctcpmlm_rate_pred(args):
    args.num_rate_level = safe_getattr(args, "num_rate_level", 5)
    args.pseudo_rate_start_step = safe_getattr(args, "pseudo_rate_start_step", 10000)
    args.initial_target_rate_level = safe_getattr(args, "initial_target_rate_level", 3)
    
    base_architecture(args)        