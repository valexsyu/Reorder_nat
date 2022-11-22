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



from fairseq.modules import (
    PositionalEmbedding,
)

from fairseq.utils import safe_getattr, safe_hasattr
logger = logging.getLogger(__name__)

from transformers import AutoModel, AutoModelForMaskedLM, AutoConfig
from torch.distributions import Categorical
import numpy as np
from scipy.optimize import linear_sum_assignment as lsa
from fairseq.utils import new_arange


@register_model("nat_pretrained_model") 
class NATPretrainedModel(BaseFairseqModel):
    def __init__(self, args, translator, src_dict, tgt_dict):
        super().__init__()
        self.translator = translator
        self.src_dict = src_dict
        self.pad = self.src_dict.pad_index
        try :
            self.mask = src_dict.indices["[MASK]"]
        except:
            self.mask = src_dict.indices["<mask>"]
        self.label_smoothing = args.label_smoothing
        self.num_upsampling_rate = args.num_upsampling_rate
        if self.num_upsampling_rate >= 10 :
            self.num_upsampling_rate = self.num_upsampling_rate / 10
        self.use_pretrained_embedding = args.use_pretrained_embedding
        self.use_drop_embedding = 1    
        self.lm_loss = args.lm_loss 
        # self.lm_loss_dis = args.lm_loss_dis
        self.pretrained_model_name = args.pretrained_model_name 
        self.pretrained_embedding_name = args.pretrained_embedding_name
        # self.pretrained_lm_name = args.pretrained_lm_name
        # self.lm_loss_layer = args.lm_loss_layer 
        self.lm_tr_layer = args.lm_tr_layer 
        self.lm_st_layer = args.lm_st_layer
        self.upsample_fill_mask = args.upsample_fill_mask
        self.dynamic_upsampling = args.dynamic_upsampling 
        self.dynamic_rate=args.dynamic_rate
        self.debug = args.debug
        if self.dynamic_rate :
            self.dynamic_upsampling = True
        if len(self.lm_st_layer) != len(self.lm_tr_layer):
            print("length of KD layer of student and teacher are not the same ")
            import pdb;pdb.set_trace()
        # if self.pretrained_lm_name is None :
        #     self.pretrained_lm = None
        # else:
        #     if self.lm_loss_dis :
        #         self.pretrained_lm = AutoModelForMaskedLM.from_pretrained(self.pretrained_lm_name)
        #     else:
        #         self.pretrained_lm = AutoModel.from_pretrained(self.pretrained_lm_name) 
        #     for params in self.pretrained_lm.parameters() :
        #         params.requires_grad=False                
        #     self.pretrained_lm.eval() 
            

        if self.pretrained_embedding_name is None:
            self.pretrained_embedding = None
        else:
            self.pretrained_embedding = AutoModel.from_pretrained(self.pretrained_embedding_name)

        # self.max_update = args.max_update
        # self.num_translation_update = args.num_translation_update
        self.lm_start_step = args.lm_start_step
        self.traning_process = "translation"
        self.do_lm_loss = False
        self.lm_head_frozen = args.lm_head_frozen
        self.embedding_frozen = args.embedding_frozen
        self.insert_position = args.insert_position
        self.lm_loss_type = args.lm_loss_type
        self.no_atten_mask = args.no_atten_mask
        if self.lm_head_frozen :
            if self.pretrained_model_name  == "distilbert-base-multilingual-cased" :
                lm_head = [self.translator.vocab_transform, self.translator.vocab_projector, self.translator.vocab_layer_norm]
            elif self.pretrained_model_name  == "bert-base-multilingual-uncased" :
                lm_head = [self.translator.cls]
            elif self.pretrained_model_name  == "jhu-clsp/bibert-ende" :
                lm_head = [self.translator.lm_head]
            elif self.pretrained_model_name  == "xlm-roberta-base" :
                lm_head = [self.translator.lm_head]
            else:
                import pdb;pdb.set_trace()
                print ("Model name Error : args.pretrained_model_name") 
            for i in range(len(lm_head)):
                for params in lm_head[i].parameters() :
                    params.requires_grad=False      

        if self.embedding_frozen:
            if self.pretrained_model_name  == "distilbert-base-multilingual-cased" :
                model_embeddings = self.translator.distilbert.embeddings
            elif self.pretrained_model_name  == "bert-base-multilingual-uncased" :
                model_embeddings = self.translator.bert.embeddings
            elif self.pretrained_model_name  == "jhu-clsp/bibert-ende" :
                model_embeddings = self.translator.roberta.embeddings       
            elif self.pretrained_model_name  == "xlm-roberta-base" :
                model_embeddings = self.translator.roberta.embeddings                                            
            else:
                import pdb;pdb.set_trace()
                print ("Model name Error : args.pretrained_model_name") 
            for params in model_embeddings.parameters() :
                params.requires_grad=False       




                                                      

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--use-align-position",
            action="store_true",
            help="input of position is discrite number if set else continues number(embed)",
        )    
       
        parser.add_argument(
            "--num-upsampling-rate",
            type=float,
            default=2,
            help="The multiplier value of the source upsampling",
        )  
        parser.add_argument(
            "--init-translator",
            action="store_true",
            help="init trnaslator para",
        )     
        parser.add_argument(
            "--use-pretrained-embedding",
            action="store_true",
            help="Use LM output to be model input embedding",
        )                 
        parser.add_argument("--pretrained-model-name", default=None, type=str,
                    help="Name of the path for the pre-trained model"
        )     
        # # args for pretrained models:
        # parser.add_argument("--pretrained-lm-name", default=None, type=str,
        #             help="Name of the path for the LM model"
        # )   
        parser.add_argument("--pretrained-embedding-name", default=None, type=str,
                    help="Name of the path for the embedding model"
        )                      
        parser.add_argument(
            "--lm-loss",
            action="store_true",
            help="compute LM loss ",
        )    
        # parser.add_argument(
        #     "--lm-loss-dis",
        #     action="store_true",
        #     help="compute LM loss using distribution ",
        # )            
        
        parser.add_argument(
            "--reorder-arch-small",
            action="store_true",
            help="reorder arch is smaller ",
        )   
        parser.add_argument(
            "--num-translation-update",
            type=int,
            default=100000,
            help="nunber of translation update",
        )     
        parser.add_argument(
            "--lm-start-step",
            type=int,
            default=75000,
            help="the step of lm loss start to update",
        )         
        # parser.add_argument(
        #     "--lm-start-step",
        #     type=int,
        #     default=75000,
        #     help="the step of lm loss start to update",
        # )            
        parser.add_argument(
            "--lm-head-frozen",
            action="store_true",
            help="Language head of model is frozen ",
        )   
        parser.add_argument(
            "--embedding-frozen",
            action="store_true",
            help="embedding of model is frozen",
        )        
        # parser.add_argument(
        #     "--lm-loss-self",
        #     action="store_true",
        #     help="self lm loss",
        # )            
        parser.add_argument(
            "--lm-loss-type",
            type=str,
            default="COS",
            help="COS or MSE",
        )                          
        # parser.add_argument(
        #     "--lm-loss-layer",
        #     type=int,
        #     default=-1,
        #     help="the lm loss layer , default is -1 (-1 means last layer)",
        # )                
        parser.add_argument(
            "--lm-st-layer",
            type=int,
            nargs="+",
            default=[-1],
            help="the lm loss layer , default is -1 (-1 means last layer)",
        )   
        parser.add_argument(
            "--lm-tr-layer",
            type=int,
            nargs="+",
            default=[-1],
            help="the lm loss layer , default is -1 (-1 means last layer)",
        )           
        parser.add_argument(
            "--upsample-fill-mask",
            action="store_true",
            help="upsample use mask to be token ",
        )       
        parser.add_argument(
            "--dynamic-upsampling",   #float rate with clipping
            action="store_true",
            help="upsample use dynamic upsampling (float rate with clipping) ",
        )       
        parser.add_argument(
            "--insert-position",
            type=str,
            default="uniform",
            help="uniform/right/left",
        )    
        parser.add_argument(
            "--dynamic-rate",        
            action="store_true",
            help="upsample use dynamic rate with dynamic upsampling if the length after upsampling is > 512   ",
        )         
        
        parser.add_argument(
            "--dropout",
            type=float,
            default=0.1,
            help="dropout",
        )              
                     
       
                    
        


    @classmethod
    def build_model(cls, args, task):
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
                   
        translator_config = AutoConfig.from_pretrained(args.pretrained_model_name)
        if not safe_hasattr(translator_config, "hidden_dropout_prob"):
            translator_config.dropout = args.dropout
        else:
            translator_config.hidden_dropout_prob = args.dropout
        
        translator = AutoModelForMaskedLM.from_pretrained(args.pretrained_model_name, config=translator_config)
        if args.init_translator :
            translator.apply(init_bert_params)  


        if OmegaConf.is_config(args):
            OmegaConf.set_struct(args, True)

        if task.cfg.no_atten_mask:
            vars(args)['no_atten_mask'] = task.cfg.no_atten_mask  
            vars(args)['debug'] = task.cfg.debug          

        return cls(args, translator, task.source_dictionary, task.target_dictionary )

               
    def forward(
        self, src_tokens, src_lengths, tgt_tokens, alignments, update_num,
        pretrained_lm=None, lm_loss_layer=-1, **kwargs
    ):  
        # if self.lm_loss and update_num > self.num_translation_update*0.75 :
        if self.lm_loss and update_num > self.lm_start_step :
            self.do_lm_loss = True
        else:
            self.do_lm_loss = False
            

        logits, output_hidden_states, rate = self.translation(src_tokens, src_lengths, **kwargs)
        src_upsample_tokens, rate = self.upsampling(src_tokens, rate)
        result = {
            "word_ins": {
                "out": logits,
                "tgt": tgt_tokens,
                "mask": None if src_upsample_tokens is None else src_upsample_tokens.ne(self.pad),
                "num_upsampling_rate": rate,
                "ls": self.label_smoothing,
                "nll_loss": True,
                "loss_type": "CTC",
            }
        }
        if pretrained_lm is not None :
            with torch.no_grad():
                target_token_embeddings, target_bos_embeddings = \
                                  self.get_pretrained_embedding(tgt_tokens, pretrained_lm, lm_loss_layer)            
            
            if self.do_lm_loss :        
                lm_loss_output = self.compute_lm_rep_loss(output_rep=output_hidden_states, logits=logits, src_tokens=src_upsample_tokens, \
                                        tgt_tokens=tgt_tokens, tgt_output_rep=target_token_embeddings, \
                                        reduce=True)  
            else :
                with torch.no_grad():
                    lm_loss_output = self.compute_lm_rep_loss(output_rep=output_hidden_states, logits=logits, src_tokens=src_upsample_tokens, \
                                            tgt_tokens=tgt_tokens, tgt_output_rep=target_token_embeddings, \
                                            reduce=True) 
            
            result.update(
                {
                    "lm": {
                        "loss": lm_loss_output,
                        "factor": 1.0,
                        "loss_type": "LOSS",
                    }  
                }                            
            )
        
        return result
                                                  

                                                        
    
    

    def lm_replace(self, output_rep, logits, src_tokens, tgt_tokens, tgt_output_rep, reduce=True):
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
            for i, x in enumerate(match_index):
                output_rep[i,x] = tgt_output_rep[i]
                
            output = self.translator.cls(output_rep)    
            
        
        return output    ### representation
         
    def forward_inference(self, src_tokens, tgt_tokens,src_lengths, alignments=None, update_num=None, **kwargs):
        
        # # ======================replace target ================================
        #     logits, output_hidden_states = self.translation(src_tokens, src_lengths, **kwargs)
        #     src_upsample_tokens = self.upsampling(src_tokens, self.num_upsampling_rate) 
        #     target_token_embeddings, target_bos_embeddings = self.get_pretrained_embedding(tgt_tokens, self.pretrained_lm, self.lm_loss_layer)            
        #     logits = self.lm_replace(output_rep=output_hidden_states, logits=logits, src_tokens=src_upsample_tokens, \
        #                             tgt_tokens=tgt_tokens, tgt_output_rep=target_token_embeddings, \
        #                             reduce=True)          
        
        logits, output_hidden_states, _= self.translation(src_tokens, src_lengths, **kwargs) 

        _scores, _tokens = F.log_softmax(logits, dim=-1).max(-1)      

        return DataOut(
            output_tokens=_tokens,
            output_scores=_scores,
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

        def compute_lm_loss(output_rep, logits, src_tokens, tgt_tokens, tgt_output_rep, reduce=True):
            bs, rep_seq_len ,_= output_rep.size()
            _, tgt_seq_len = tgt_tokens.size()
            target = tgt_tokens.repeat(1, rep_seq_len).view(bs, rep_seq_len, tgt_seq_len)
            bipart_no_pad = target.ne(self.pad)
            src_no_pad = src_tokens.ne(self.pad)
            bipart_lprobs = F.log_softmax(logits, dim=-1)
            nll_loss = -bipart_lprobs.gather(dim=-1, index=target)#bs rep_seq_len tgt_seq_len
            nll_loss = nll_loss * bipart_no_pad

            nll_loss_numpy = nll_loss.detach()
            tgt_output_rep = tgt_output_rep.detach()
            lm_loss = torch.zeros(1).to(src_tokens.device)
            for batch_id in range(bs):
                no_pad_num = bipart_no_pad[batch_id, 0].sum()
                src_no_pad_num = src_no_pad[batch_id].sum()
                output_tokens = logits[batch_id].argmax(-1)
                output_tokens_blank_mask = output_tokens.eq(self.src_dict.bos()).view(-1,1).repeat(1,tgt_seq_len)
                nll_loss_numpy_line = nll_loss_numpy[batch_id]
                nll_loss_numpy_line = nll_loss_numpy_line.masked_fill_(output_tokens_blank_mask, float(10^8))
                raw_index, col_index = lsa(nll_loss_numpy_line[:src_no_pad_num, :no_pad_num].cpu().numpy())
                lm_loss = ((1 - F.cosine_similarity(output_rep[batch_id][raw_index], tgt_output_rep[batch_id][col_index])).mean())+ lm_loss
                
            return lm_loss/bs

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
        if self.lm_loss_type == "COS" :
            output_lm_loss = 1 - F.cosine_similarity(match_output_rep, tgt_output_rep, dim=2).mean()
        elif self.lm_loss_type == "MSE" :
            output_lm_loss = F.mse_loss(match_output_rep,tgt_output_rep,reduction='mean')
        else:
            raise NotImplementedError
            
        
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

    def get_pretrained_embedding(self, src_tokens, model, output_layer=-1) :
        device = src_tokens.device
        model.to(device) 
        bos = self.src_dict.bos() * torch.ones(src_tokens.shape[0], 1, dtype=torch.long, device=device)
        src_tokens = torch.cat((bos, src_tokens), dim=1)
        lm_outputs = model(src_tokens, output_hidden_states=True, return_dict=True) 
        token_embeddings = lm_outputs['hidden_states'][output_layer].detach()
        # random_num = torch.rand(1)
        # token_embeddings = token_embeddings[-(int(random_num * self.use_drop_embedding)+1)]     
        # token_embeddings = token_embeddings[-1].detach()    
        bos_embeddings = token_embeddings[:, 0, :].detach().unsqueeze(1)
        token_embeddings = token_embeddings[:, 1:, :].detach()   
        src_tokens = src_tokens[:, 1:]    

        return token_embeddings, bos_embeddings

    def get_pretrained_lprobs(self, src_tokens, model) :
        device = src_tokens.device
        model.to(device)
        bos = self.src_dict.bos() * torch.ones(src_tokens.shape[0], 1, dtype=torch.long, device=device)
        src_tokens = torch.cat((bos, src_tokens), dim=1)
        lm_outputs = model(src_tokens, output_hidden_states=False, return_dict=True)
        logits = lm_outputs["logits"]
        lm_lprobs = F.log_softmax(logits[:, 1:, :].detach(),-1) 
        src_tokens = src_tokens[:, 1:]                
        
        return lm_lprobs
    def upsampling(self, source, rate):          
        def dynamic_upsample_token(x, insert_mask=False , rate=2, insertion_position='uniform'):
            B, L = x.size(0), x.size(1)
            new_length = torch.Tensor([L * rate]).int().item()
            # new_length = int(L * rate)   # 50*2.3=114.9999999
            pad = self.src_dict.pad()
            bos = self.src_dict.bos()
            eos = self.src_dict.eos()
            ### the mask is True when padding/bos/eos 
            mask = ~(
                x.ne(pad) & x.ne(bos) & x.ne(eos)
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

                new_location = t_w.argmax(-1)
                
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
                
                t_x[torch.where(t_x == pad)] = self.mask
                
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
                         


        if self.upsample_fill_mask :
            if self.dynamic_upsampling :
                if self.dynamic_rate :
                    B, L = source.size(0), source.size(1)
                    new_length = torch.Tensor([L * rate]).int().item()    
                    if new_length > self.translator.config.max_position_embeddings :
                        rate= float(self.translator.config.max_position_embeddings)/float(L)
                    
                insert_mask = True
                t_x, t_mask, w, t_w, new_t_w, new_location = dynamic_upsample_token(source, insert_mask , rate, insertion_position=self.insert_position)  
                return t_x, rate  
            else :
                b,l = source.size()
                mask = source.ne(self.pad)  #ex : soruce=[7,7,7,pad] mask=[True True True False]
                mask_tokens = source.masked_fill(mask,self.mask)
                if rate == 5 :
                    upsampled = torch.stack((source, mask_tokens, mask_tokens, mask_tokens, mask_tokens), dim=2).view(b, int(l*rate))                   
                elif rate == 4 :
                    upsampled = torch.stack((source, mask_tokens, mask_tokens, mask_tokens), dim=2).view(b, int(l*rate))                
                elif rate == 3 :
                    upsampled = torch.stack((source, mask_tokens, mask_tokens), dim=2).view(b, int(l*rate))
                elif rate == 2:
                    upsampled = torch.stack((source, mask_tokens), dim=2).view(b, int(l*rate))
                else:
                    print("Not support the rate in this setting")
                    import pdb;pdb.set_trace()
        else:    
            if self.dynamic_upsampling :
                if self.dynamic_rate :
                    B, L = source.size(0), source.size(1)
                    new_length = torch.Tensor([L * rate]).int().item()    
                    if new_length > self.translator.config.max_position_embeddings :
                        rate= float(self.translator.config.max_position_embeddings)/float(L)                
                insert_mask = False
                t_x, t_mask, w, t_w, new_t_w, new_location = dynamic_upsample_token(source, insert_mask , rate) 
                return t_x, rate 
            else:                        
                upsampled =  torch.repeat_interleave(source, int(rate), dim=1)
        return upsampled, rate

    def translation(self, src_tokens, src_lengths, **kwargs):
        bos_embeddings = None
        if self.use_pretrained_embedding :
            with torch.no_grad():
                token_embeddings, bos_embeddings = self.get_pretrained_embedding(src_tokens, self.pretrained_embedding)
            translator_token_embedding = torch.cat((bos_embeddings, translator_token_embedding), dim=1)
            output_translator = self.translator.forward(input_ids = None, output_hidden_states=True, return_dict=True, 
                                    inputs_embeds=translator_token_embedding)                
                
        else:       
        # forget to use bos to training
            bos = self.src_dict.bos() * torch.ones(src_tokens.shape[0], 1, dtype=torch.long, device=src_tokens.device)
            src_tokens_upsample, rate = self.upsampling(src_tokens, self.num_upsampling_rate)
            src_tokens_upsample = torch.cat((bos, src_tokens_upsample), dim=1)  
            atttention_mask=src_tokens_upsample.ne(self.pad)
            if self.no_atten_mask :
                output_translator = self.translator.forward(input_ids = src_tokens_upsample, 
                                    output_hidden_states=True, return_dict=True, 
                                        inputs_embeds=None)                 
            else:
                output_translator = self.translator.forward(input_ids = src_tokens_upsample, 
                                attention_mask=atttention_mask,  #encoder_attention_mask=atttention_mask,
                                output_hidden_states=True, return_dict=True, 
                                    inputs_embeds=None)     
        logits = output_translator['logits'][:,1:,:]
        hidden_states = output_translator['hidden_states'][-1][:,1:,:]

        #     if self.pretrained_model_name is not None:
        #         if self.pretrained_model_name  == "distilbert-base-multilingual-cased" :
        #            token_embeddings = self.translator.distilbert.embeddings.word_embeddings(src_tokens)
        #         elif self.pretrained_model_name  == "bert-base-multilingual-uncased" :
        #             token_embeddings = self.translator.bert.embeddings.word_embeddings(src_tokens)
        #         elif self.pretrained_model_name  == "jhu-clsp/bibert-ende" :
        #             token_embeddings = self.translator.roberta.embeddings.word_embeddings(src_tokens)
        #         else:
        #             import pdb;pdb.set_trace()
        #             print ("Model Error")
                
        #     else:
        #         token_embeddings = self.translator.sentence_encoder.embed_tokens(src_tokens) 
        #         import pdb;pdb.set_trace()
            
        # translator_token_embedding = self.upsampling(token_embeddings, self.num_upsampling_rate)
        # if bos_embeddings is not None:
        #     translator_token_embedding = torch.cat((bos_embeddings, translator_token_embedding), dim=1)
        # output_translator = self.translator.forward(input_ids = None, output_hidden_states=True, return_dict=True, 
        #                             inputs_embeds=translator_token_embedding)
        # logits = output_translator['logits']
        # hidden_states = output_translator['hidden_states'][-1]
        
        # if bos_embeddings is not None :
        #     logits = logits[:,1:,:]
        #     hidden_states = hidden_states[:,1:,:]

        return logits, hidden_states , rate
             

@register_model_architecture(
    "nat_pretrained_model", "nat_pretrained_model"
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
    # args.lm_loss_dis  = safe_getattr( args, "lm_loss_dis", False )
    args.lm_head_frozen  = safe_getattr( args, "lm_head_frozen", False )
    args.embedding_frozen  = safe_getattr( args, "embedding_frozen", False )
    args.lm_loss_self  = safe_getattr( args, "lm_loss_self", False )
    args.lm_loss_type  = safe_getattr( args, "lm_loss_type", "COS" )
    # args.lm_loss_layer  = safe_getattr( args, "lm_loss_layer", -1 )
    args.lm_tr_layer  = safe_getattr( args, "lm_tr_layer", [-1] )
    args.lm_st_layer  = safe_getattr( args, "lm_st_layer", [-1] )
    args.lm_start_step = safe_getattr( args, "lm_start_step", 75000)
    
    args.upsample_fill_mask  = safe_getattr( args, "upsample_fill_mask", False )
    args.dynamic_upsampling  = safe_getattr( args, "dynamic_upsampling", False )
    args.dynamic_rate = safe_getattr( args, "dynamic_rate", False )
    args.insert_position  = safe_getattr( args, "insert_position", "uniform" )
    
    args.no_atten_mask = safe_getattr(args, "no_atten_mask", False )
    args.debug = safe_getattr(args, "debug", False )
    
    
    


    # Trsanslator config
    args.max_translator_positions = safe_getattr(args, "max_translator_positions", 1024)
    # Reorder_Translation config
    args.reorder_translation = safe_getattr(args, "reorder_translation", "reorder_translation")    
    args.max_align_positions = safe_getattr(args, "max_align_positions", 512)
    args.num_upsampling_rate = safe_getattr(args, "num_upsampling_rate", 2 )
    args.use_pretrained_embedding = safe_getattr(args, "use_pretrained_embedding", False )
    args.pretrained_model_name = safe_getattr(args, "pretrained_model_name", None )
    args.pretrained_embedding_name = safe_getattr(args, "pretrained_embedding_name", None )
    # args.pretrained_lm_name = safe_getattr(args, "pretrained_lm_name", None )
    args.reorder_arch_small = safe_getattr(args, "reorder_arch_small", False )
    
    
@register_model_architecture(
    "nat_pretrained_model", "nat_pretrained_model_samll"
)
def small_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 6)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 6)
    base_architecture(args)

@register_model_architecture(
    "nat_pretrained_model", "nat_pretrained_model_medium"
)
def medium_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 8)
    base_architecture(args)
