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
from fairseq.models import (
    BaseFairseqModel,
)

from fairseq.models.roberta import (
    RobertaEncoder,
    RobertaModel,
    RobertaLMHead,
)

from fairseq.modules import (
    PositionalEmbedding,
)

from fairseq.utils import safe_getattr, safe_hasattr
logger = logging.getLogger(__name__)

from transformers import AutoModel, AutoModelForMaskedLM
from torch.distributions import Categorical
import numpy as np
from scipy.optimize import linear_sum_assignment as lsa


@register_model("nat_position_reorder") 
class NATPositionalReorderModel(BaseFairseqModel):
    def __init__(self, args, translator_encoder, reorder_encoder, src_dict, tgt_dict):
        super().__init__()
        self.translator_encoder = translator_encoder
        self.reorder_encoder = reorder_encoder
        self.src_dict = src_dict
        if reorder_encoder is not None :
            self.embed_align_positions = PositionalEmbedding(
                    args.max_align_positions,
                    args.encoder_embed_dim,
                    self.src_dict.pad_index,
                    learned=True,
                )
        self.pad = self.src_dict.pad_index
        self.use_align_position = args.use_align_position
        self.label_smoothing = args.label_smoothing
        self.num_upsampling_rate = args.num_upsampling_rate
        self.use_pretrained_embedding = args.use_pretrained_embedding
        self.use_drop_embedding = 1    
        self.lm_loss = args.lm_loss 
        self.lm_loss_dis = args.lm_loss_dis
        self.pretrained_model_name = args.pretrained_model_name 
        self.pretrained_embedding_name = args.pretrained_embedding_name
        self.pretrained_lm_name = args.pretrained_lm_name
        if self.pretrained_lm_name is None :
            self.pretrained_lm = None
        else:
            if self.lm_loss_dis :
                self.pretrained_lm = AutoModelForMaskedLM.from_pretrained(self.pretrained_lm_name)
            else:
                self.pretrained_lm = AutoModel.from_pretrained(self.pretrained_lm_name) 
            self.pretrained_lm.eval()        
        # # 7-6-1 7-9-1
        if self.pretrained_embedding_name is None or self.pretrained_embedding_name ==  self.pretrained_lm_name :
            self.pretrained_embedding = self.pretrained_lm #share parameters
        else:
            self.pretrained_embedding = AutoModel.from_pretrained(self.pretrained_embedding_name)

        # self.max_update = args.max_update
        self.num_translation_update = args.num_translation_update
        self.num_reorder_translation_update = args.num_reorder_translation_update
        self.num_reorder_update = args.num_reorder_update

        self.traning_process = "translation"
        self.do_lm_loss = False
        self.reorder_translation_state = args.reorder_translation




        


                                                      

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # RobertaModel.add_args(parser)
              
        parser.add_argument(
            "--use-align-position",
            action="store_true",
            help="input of position is discrite number if set else continues number(embed)",
        )    

        parser.add_argument(
            "--reorder-translation",
            choices=['reorder_translation', 'reorder', 'translation'],
            default="reorder_translation",
            help="choise the model type, reorder-translation/reorder/translation",
        )           
        parser.add_argument(
            "--num-upsampling-rate",
            type=int,
            default=2,
            help="The multiplier value of the source upsampling",
        )  
        parser.add_argument(
            "--init-translator",
            action="store_true",
            help="init trnaslator para",
        )     
        parser.add_argument(
            "--init-reorder",
            action="store_true",
            help="init reorder para",
        )     
        parser.add_argument(
            "--use-pretrained-embedding",
            action="store_true",
            help="Use LM output to be model input embedding",
        )                 
        parser.add_argument("--pretrained-model-name", default=None, type=str,
                    help="Name of the path for the pre-trained model"
        )     
        # args for pretrained models:
        parser.add_argument("--pretrained-lm-name", default=None, type=str,
                    help="Name of the path for the LM model"
        )   
        parser.add_argument("--pretrained-embedding-name", default=None, type=str,
                    help="Name of the path for the embedding model"
        )                      
        parser.add_argument(
            "--lm-loss",
            action="store_true",
            help="compute LM loss ",
        )    
        parser.add_argument(
            "--lm-loss-dis",
            action="store_true",
            help="compute LM loss using distribution ",
        )            
        
        parser.add_argument(
            "--reorder-arch-small",
            action="store_true",
            help="reorder arch is smaller ",
        )   
        parser.add_argument(
            "--num-translation-update",
            type=int,
            default=50000,
            help="nunber of translation update",
        )     
        parser.add_argument(
            "--num-reorder-update",
            type=int,
            default=0,
            help="nunber of translation update",
        )   
        parser.add_argument(
            "--num-reorder-translation-update",
            type=int,
            default=0,
            help="nunber of reorder-translation update",
        )       
        # parser.add_argument(
        #     "--max-update",
        #     type=int,
        #     default=100000,
        #     help="nunber of training update",
        # )                         
        


    @classmethod
    def build_model(cls, args, task):
        def small_arch(args):
            args.encoder_layers = 6
            args.encoder_embed_dim = 768
            args.encoder_ffn_embed_dim = 2048
            args.encoder_attention_heads = 6
            return args

        """Build a new model instance."""

        from omegaconf import OmegaConf

        if OmegaConf.is_config(args):
            OmegaConf.set_struct(args, False)

        # make sure all arguments are present
        base_architecture(args)
        if not safe_hasattr(args, "max_positions"):
            if not safe_hasattr(args, "tokens_per_sample"):
                args.tokens_per_sample = task.max_positions()
            args.max_positions = args.tokens_per_sample
            
        if args.reorder_translation == 'translation' and args.use_align_position == False :
            reorder_encoder = None
        else:
            reorder_encoder = AutoModel.from_pretrained(args.pretrained_model_name)
            if args.init_reorder :
                self.reorder_encoder.apply(init_bert_params)           
        translator_encoder = AutoModelForMaskedLM.from_pretrained(args.pretrained_model_name)
        if args.init_translator :
            translator_encoder.apply(init_bert_params)            
        

        if OmegaConf.is_config(args):
            OmegaConf.set_struct(args, True)

        return cls(args, translator_encoder, reorder_encoder, task.source_dictionary, task.target_dictionary )

               
    def forward(
        self, src_tokens, src_lengths, tgt_tokens, alignments, update_num, **kwargs
    ):  
        if update_num <= self.num_translation_update :
            self.traning_process = "translation"
            if self.lm_loss and update_num > self.num_translation_update / 2 :
                self.do_lm_loss = True
            else:
                self.do_lm_loss = False
        elif update_num <= self.num_translation_update + self.num_reorder_update :
            self.traning_process = "reorder"
        else:
            self.traning_process = "reorder_translation"
            if self.lm_loss :
                self.do_lm_loss = True
            else:
                self.do_lm_loss = False




        if self.traning_process == "translation" :
            logits, output_hidden_states = self.translation(src_tokens, src_lengths, alignments, **kwargs)
            src_upsample_tokens = self.upsampling(src_tokens, self.num_upsampling_rate)
            if self.do_lm_loss :   
                if self.lm_loss_dis :
                    with torch.no_grad():
                        tgt_output_lprobs=self.get_pretrained_lprobs(tgt_tokens)
                    lm_loss_output = self.compute_lm_dis_loss(logits, src_tokens=src_upsample_tokens, tgt_tokens=tgt_tokens, \
                                                tgt_output_lprobs=tgt_output_lprobs, reduce=True)
                else:
                    with torch.no_grad():
                        target_token_embeddings, target_bos_embeddings = self.get_pretrained_embedding(tgt_tokens)
                    lm_loss_output = self.compute_lm_rep_loss(output_rep=output_hidden_states, logits=logits, src_tokens=src_upsample_tokens, \
                                            tgt_tokens=tgt_tokens, tgt_output_rep=target_token_embeddings, \
                                            reduce=True)  

                return {
                    "word_ins": {
                        "out": logits,
                        "tgt": tgt_tokens,
                        "mask": None if src_upsample_tokens is None else src_upsample_tokens.ne(self.pad),
                        "num_upsamling_rate": self.num_upsampling_rate,
                        "ls": self.label_smoothing,
                        "nll_loss": True,
                        "loss_type": "CTC",
                    },
                    "lm": {
                        "loss": lm_loss_output,
                        "factor": 1.0,
                        "loss_type": "LOSS",
                    },                
                }                                                                 
            return {
                "word_ins": {
                    "out": logits,  
                    "tgt": tgt_tokens,
                    "mask": None if src_upsample_tokens is None else src_upsample_tokens.ne(self.pad),
                    "num_upsamling_rate": self.num_upsampling_rate,
                    "ls": self.label_smoothing,
                    "nll_loss": True,
                    "loss_type": "CTC",
                },     
            }    

        elif self.traning_process == "reorder" :
            logits, align_positions_embed = self.reorder(src_tokens, src_lengths, alignments, **kwargs)
            return {
                "reorder_ins": {
                    "out": logits, 
                    "tgt": align_positions_embed,
                    "mask": None if src_tokens is None else src_tokens.ne(self.pad),
                    "ls": self.label_smoothing,
                    "nll_loss": True,
                    "loss_type": "MSE",
                },
            }                     
            
        elif self.traning_process == "reorder_translation" : 
            logits, reorder_position_embed_out, output_hidden_states, align_positions_embed = \
                        self.reorder_translation(src_tokens, src_lengths, alignments, **kwargs)
            src_upsample_tokens = self.upsampling(src_tokens, self.num_upsampling_rate)
            if self.do_lm_loss :
                if self.lm_loss_dis :
                    with torch.no_grad():
                        tgt_output_lprobs=self.get_pretrained_lprobs(tgt_tokens)
                    lm_loss_output = self.compute_lm_dis_loss(logits, src_tokens=src_upsample_tokens, tgt_tokens=tgt_tokens, \
                                                tgt_output_lprobs=tgt_output_lprobs, reduce=True)
                else:
                    with torch.no_grad():
                        target_token_embeddings, target_bos_embeddings = self.get_pretrained_embedding(tgt_tokens)
                    lm_loss_output = self.compute_lm_rep_loss(output_rep=output_hidden_states, logits=logits, src_tokens=src_upsample_tokens, \
                                            tgt_tokens=tgt_tokens, tgt_output_rep=target_token_embeddings, \
                                            reduce=True)                                                  

                return {
                    "word_ins": {
                        "out": logits,
                        "tgt": tgt_tokens,
                        "mask": None if src_upsample_tokens is None else src_upsample_tokens.ne(self.pad),
                        "num_upsamling_rate": self.num_upsampling_rate,
                        "ls": self.label_smoothing,
                        "nll_loss": True,
                        "loss_type": "CTC",
                    },
                    "reorder_ins": {
                        "out": reorder_position_embed_out,
                        "tgt": align_positions_embed,
                        "mask": None if src_tokens is None else src_tokens.ne(self.pad),
                        "ls": self.label_smoothing,
                        "nll_loss": True,
                        "factor": 0.001,
                        "loss_type": "MSE",
                    },
                    "lm": {
                        "loss": lm_loss_output,
                        "factor": 1.0,
                        "loss_type": "LOSS",
                    },                
                }      

            return {
                "word_ins": {
                    "out": logits,
                    "tgt": tgt_tokens,
                    "mask": None if src_upsample_tokens is None else src_upsample_tokens.ne(self.pad),
                    "num_upsamling_rate": self.num_upsampling_rate,
                    "ls": self.label_smoothing,
                    "nll_loss": True,
                    "loss_type": "CTC",
                },
                "reorder_ins": {
                    "out": reorder_position_embed_out,
                    "tgt": align_positions_embed,
                    "mask": None if src_tokens is None else src_tokens.ne(self.pad),
                    "ls": self.label_smoothing,
                    "nll_loss": True,
                    "loss_type": "MSE",
                },            
            }               
         
    def forward_inference(self, src_tokens, src_lengths, alignments, update_num=None, **kwargs):
        if self.reorder_translation_state == "translation" :
            logits, output_hidden_states = self.translation(src_tokens, src_lengths, alignments, **kwargs)                
        elif self.reorder_translation_state == "reorder_translation" :
            logits, reorder_position_embed_out, output_hidden_states, align_positions_embed = \
                        self.reorder_translation(src_tokens, src_lengths, alignments, **kwargs)
                  

        probs = F.softmax(logits,-1)
        # entropys = Categorical(probs).entropy()
        token_probs, _tokens = probs.max(-1)
        _scores = torch.log(token_probs)              

        return DataOut(
            output_tokens=_tokens,
            output_scores=_scores,
            # output_entropys=entropys,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )  

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)      

        # def compute_lm_loss(output_rep, logits, src_tokens, tgt_tokens, tgt_output_rep, reduce=True):
        #     bs, rep_seq_len ,_= output_rep.size()
        #     _, tgt_seq_len = tgt_tokens.size()
        #     target = tgt_tokens.repeat(1, rep_seq_len).view(bs, rep_seq_len, tgt_seq_len)
        #     bipart_no_pad = target.ne(self.pad)
        #     src_no_pad = src_tokens.ne(self.pad)
        #     bipart_lprobs = F.log_softmax(logits, dim=-1)
        #     nll_loss = -bipart_lprobs.gather(dim=-1, index=target)#bs rep_seq_len tgt_seq_len
        #     nll_loss = nll_loss * bipart_no_pad

        #     nll_loss_numpy = nll_loss.detach()
        #     tgt_output_rep = tgt_output_rep.detach()
        #     lm_loss = torch.zeros(1).to(src_tokens.device)
        #     for batch_id in range(bs):
        #         no_pad_num = bipart_no_pad[batch_id, 0].sum()
        #         src_no_pad_num = src_no_pad[batch_id].sum()
        #         output_tokens = logits[batch_id].argmax(-1)
        #         output_tokens_blank_mask = output_tokens.eq(self.src_dict.bos()).view(-1,1).repeat(1,tgt_seq_len)
        #         nll_loss_numpy_line = nll_loss_numpy[batch_id]
        #         nll_loss_numpy_line = nll_loss_numpy_line.masked_fill_(output_tokens_blank_mask, float(10^8))
        #         raw_index, col_index = lsa(nll_loss_numpy_line[:src_no_pad_num, :no_pad_num].cpu().numpy())
        #         lm_loss = ((1 - F.cosine_similarity(output_rep[batch_id][raw_index], tgt_output_rep[batch_id][col_index])).mean())+ lm_loss
                
        #     return lm_loss/bs

    def compute_lm_rep_loss(self, output_rep, logits, src_tokens, tgt_tokens, tgt_output_rep, reduce=True):
        with torch.no_grad():
            bs, rep_seq_len ,_= output_rep.size()
            _, tgt_seq_len = tgt_tokens.size()
            target = tgt_tokens.repeat(1, rep_seq_len).view(bs, rep_seq_len, tgt_seq_len)
            bipart_no_pad = target.ne(self.pad)
            src_no_pad = src_tokens.ne(self.pad)
            bipart_lprobs = F.log_softmax(logits, dim=-1)
            nll_loss = -bipart_lprobs.gather(dim=-1, index=target)#bs rep_seq_len tgt_seq_len
            nll_loss = nll_loss * bipart_no_pad
            match_index = nll_loss.argmin(1)
        match_output_rep = output_rep[torch.arange(output_rep.shape[0]).unsqueeze(-1), match_index]
        output_lm_loss = 1 - F.cosine_similarity(match_output_rep, tgt_output_rep, dim=2).mean()
        
        return output_lm_loss    ### representation

    def compute_lm_dis_loss(self, logits, src_tokens, tgt_tokens, tgt_output_lprobs, reduce=True):
        with torch.no_grad():
            bs, rep_seq_len ,_ = logits.size()
            _, tgt_seq_len = tgt_tokens.size()
            target = tgt_tokens.repeat(1, rep_seq_len).view(bs, rep_seq_len, tgt_seq_len)
            bipart_no_pad = target.ne(self.pad)
            src_no_pad = src_tokens.ne(self.pad)
            bipart_lprobs = F.log_softmax(logits, dim=-1)
            tokens_lprob = bipart_lprobs.gather(dim=-1, index=target)#bs rep_seq_len tgt_seq_len
            tokens_lprob = tokens_lprob * bipart_no_pad
            match_index = tokens_lprob.argmax(1)
        match_output_lprobs = bipart_lprobs[torch.arange(logits.shape[0]).unsqueeze(-1), match_index]
        output_lm_loss = F.kl_div(match_output_lprobs, tgt_output_lprobs.to(match_output_lprobs.device),
                                    reduction="batchmean", log_target=True)
        return output_lm_loss  

    def get_pretrained_embedding(self, src_tokens) :
        device = src_tokens.device
        self.pretrained_embedding.to(device) 
        bos = self.src_dict.bos() * torch.ones(src_tokens.shape[0], 1, dtype=torch.long, device=device)
        src_tokens = torch.cat((bos, src_tokens), dim=1)
        lm_outputs = self.pretrained_embedding(src_tokens, output_hidden_states=True, return_dict=True) 
        token_embeddings = lm_outputs['hidden_states'][-1].detach()
        # random_num = torch.rand(1)
        # token_embeddings = token_embeddings[-(int(random_num * self.use_drop_embedding)+1)]     
        # token_embeddings = token_embeddings[-1].detach()    
        bos_embeddings = token_embeddings[:, 0, :].detach().unsqueeze(1)
        token_embeddings = token_embeddings[:, 1:, :].detach()   
        src_tokens = src_tokens[:, 1:]    

        return token_embeddings, bos_embeddings

    def get_pretrained_lprobs(self, src_tokens) :
        device = src_tokens.device
        self.pretrained_lm.to(device)
        bos = self.src_dict.bos() * torch.ones(src_tokens.shape[0], 1, dtype=torch.long, device=device)
        src_tokens = torch.cat((bos, src_tokens), dim=1)
        lm_outputs = self.pretrained_lm(src_tokens, output_hidden_states=False, return_dict=True)
        logits = lm_outputs["logits"]
        lm_lprobs = F.log_softmax(logits[:, 1:, :].detach(),-1) 
        src_tokens = src_tokens[:, 1:]                
        
        return lm_lprobs
    def upsampling(self, source, rate): 
        upsampled =  torch.repeat_interleave(source, rate, dim=1)
        return upsampled

    def translation(self, src_tokens, src_lengths, alignments, **kwargs):
        bos_embeddings = None
        if self.use_pretrained_embedding :
            with torch.no_grad():
                token_embeddings, bos_embeddings = self.get_pretrained_embedding(src_tokens)
        else:       
            if self.pretrained_model_name is not None:
                token_embeddings = self.translator_encoder.roberta.embeddings.word_embeddings(src_tokens)
            else:
                token_embeddings = self.translator_encoder.sentence_encoder.embed_tokens(src_tokens) 

        if self.use_align_position :
            align_positions_embed = self.embed_align_positions(alignments)
            translator_token_embedding = token_embeddings + align_positions_embed
        else:
            translator_token_embedding = token_embeddings                
        translator_token_embedding = self.upsampling(translator_token_embedding, self.num_upsampling_rate)
        if bos_embeddings is not None:
            translator_token_embedding = torch.cat((bos_embeddings, translator_token_embedding), dim=1)

        output_translator = self.translator_encoder.forward(input_ids = None, output_hidden_states=True, return_dict=True, 
                                    inputs_embeds=translator_token_embedding)
        logits = output_translator['logits']
        hidden_states = output_translator['hidden_states'][-1]


        if bos_embeddings is not None :
            logits = logits[:,1:,:]
            hidden_states = hidden_states[:,1:,:]

        return logits, hidden_states

    def reorder(self, src_tokens, src_lengths, alignments, **kwargs):
        with torch.no_grad(): 
            align_positions_embed = self.embed_align_positions(alignments)
        bos_embeddings = None
        if self.use_pretrained_embedding :
            with torch.no_grad():
                token_embeddings, bos_embeddings = self.get_pretrained_embedding(src_tokens)
        else:
            token_embeddings = None  

        if bos_embeddings is not None:
            token_embeddings = torch.cat((bos_embeddings, token_embeddings), dim=1)

        logits, inner_state = self.reorder_encoder.forward(input_ids = None if self.use_pretrained_embedding else src_tokens,
                                    output_hidden_states=False, return_dict=False,
                                    inputs_embeds=token_embeddings)     
                                            
        if bos_embeddings is not None :
            logits = logits[:,1:,:]
                    
        return logits, align_positions_embed

    def reorder_translation(self, src_tokens, src_lengths, alignments, **kwargs):
        bos_embeddings = None
        if self.use_pretrained_embedding :
            with torch.no_grad():
                token_embeddings, bos_embeddings = self.get_pretrained_embedding(src_tokens)
        else:              
            if self.pretrained_model_name is not None:
                token_embeddings = self.translator_encoder.embeddings.word_embeddings(src_tokens)
            else:
                token_embeddings = self.translator_encoder.sentence_encoder.embed_tokens(src_tokens) 
        if bos_embeddings is not None:
            token_embeddings = torch.cat((bos_embeddings, token_embeddings), dim=1)

        logits, inner_state = self.reorder_encoder.forward(input_ids = None if self.use_pretrained_embedding else src_tokens , 
                                    output_hidden_states=False, return_dict=False, #$$
                                    inputs_embeds=token_embeddings)      

        if bos_embeddings is not None:
            reorder_position_embed_out = logits[:,1:,:] 
            token_embeddings = token_embeddings[:,1:,:]
        else:
            reorder_position_embed_out = logits

        if self.use_align_position :
            with torch.no_grad(): 
                align_positions_embed = self.embed_align_positions(alignments)   
            translator_token_embedding = token_embeddings + reorder_position_embed_out 
            # translator_token_embedding = token_embeddings + align_positions_embed ## test ground True
        else:
            translator_token_embedding = token_embeddings   
            align_positions_embed = None               
        translator_token_embedding = self.upsampling(translator_token_embedding, self.num_upsampling_rate)    
        if bos_embeddings is not None:
            translator_token_embedding = torch.cat((bos_embeddings, translator_token_embedding), dim=1)                
                
        output_translator = self.translator_encoder.forward(input_ids = None, output_hidden_states=True, return_dict=True, 
                                    inputs_embeds=translator_token_embedding)
        logits = output_translator['logits']
        hidden_states = output_translator['hidden_states'][-1]

        if bos_embeddings is not None :
            logits = logits[:,1:,:]
            hidden_states = hidden_states[:,1:,:]

        return logits, reorder_position_embed_out, hidden_states, align_positions_embed              
 

@register_model_architecture(
    "nat_position_reorder", "nat_position_reorder"
)
def base_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 12)

    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = safe_getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = safe_getattr(args, "pooler_dropout", 0.0)

    args.max_source_positions = safe_getattr(args, "max_positions", 512)
    args.no_token_positional_embeddings = safe_getattr(
        args, "no_token_positional_embeddings", False
    )

    # BERT has a few structural differences compared to the original Transformer
    args.encoder_learned_pos = safe_getattr(args, "encoder_learned_pos", True)
    args.layernorm_embedding = safe_getattr(args, "layernorm_embedding", True)
    args.no_scale_embedding = safe_getattr(args, "no_scale_embedding", True)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = safe_getattr(
        args, "encoder_normalize_before", False
    )
    args.pooler_activation_fn = safe_getattr(args, "pooler_activation_fn", "tanh")
    args.untie_weights_roberta = safe_getattr(args, "untie_weights_roberta", False)

    # Adaptive input config
    args.adaptive_input = safe_getattr(args, "adaptive_input", False)

    # LayerDrop config
    args.encoder_layerdrop = safe_getattr(args, "encoder_layerdrop", 0.0)
    args.encoder_layers_to_keep = safe_getattr(args, "encoder_layers_to_keep", None)

    # Quantization noise config
    args.quant_noise_pq = safe_getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = safe_getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = safe_getattr(args, "quant_noise_scalar", 0)

    # R4F config
    args.spectral_norm_classification_head = safe_getattr(
        args, "spectral_norm_classification_head", False
    )

    # R4F config
    args.use_align_position  = safe_getattr( args, "use_align_position", False )
    args.init_translator  = safe_getattr( args, "init_translator", False )
    args.init_reorder  = safe_getattr( args, "init_reorder", False )
    args.lm_loss  = safe_getattr( args, "lm_loss", False )
    args.lm_loss_dis  = safe_getattr( args, "lm_loss_dis", False )


    # Trsanslator config
    args.max_translator_positions = safe_getattr(args, "max_translator_positions", 1024)
    # Reorder_Translation config
    args.reorder_translation = safe_getattr(args, "reorder_translation", "reorder_translation")    
    args.max_align_positions = safe_getattr(args, "max_align_positions", 512)
    args.num_upsampling_rate = safe_getattr(args, "num_upsampling_rate", 2 )
    args.use_pretrained_embedding = safe_getattr(args, "use_pretrained_embedding", False )
    args.pretrained_model_name = safe_getattr(args, "pretrained_model_name", None )
    args.pretrained_embedding_name = safe_getattr(args, "pretrained_embedding_name", None )
    args.pretrained_lm_name = safe_getattr(args, "pretrained_lm_name", None )
    args.reorder_arch_small = safe_getattr(args, "reorder_arch_small", False )
    
    
@register_model_architecture(
    "nat_position_reorder", "nat_position_reorder_samll"
)
def small_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 6)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 6)
    base_architecture(args)

@register_model_architecture(
    "nat_position_reorder", "nat_position_reorder_medium"
)
def medium_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 8)
    base_architecture(args)