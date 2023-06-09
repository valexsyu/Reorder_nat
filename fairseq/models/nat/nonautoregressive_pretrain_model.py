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
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt   
from omegaconf import ListConfig

@register_model("nat_pretrained_model") 
class NATPretrainedModel(BaseFairseqModel):
    def __init__(self, args, translator, src_dict, tgt_dict):
        super().__init__()
        
        self.translator = translator
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.pad = self.src_dict.pad_index
        try :
            self.mask = src_dict.indices["[MASK]"]
        except:
            self.mask = src_dict.indices["<mask>"]
        self.num_upsampling_rate = args.num_upsampling_rate
        if self.num_upsampling_rate >= 10 :
            self.num_upsampling_rate = self.num_upsampling_rate / 10
        self.use_pretrained_embedding = args.use_pretrained_embedding
        self.use_drop_embedding = 1    
        self.lm_loss = args.lm_loss 
        self.lmk_loss = args.lmk_loss 
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
        self.has_eos = args.has_eos
        if self.dynamic_rate :
            self.dynamic_upsampling = True
        if len(self.lm_st_layer) != len(self.lm_tr_layer):
            print("length of KD layer of student and teacher are not the same ")
            import pdb;pdb.set_trace()
         

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
        self.lm_mask_rate = args.lm_mask_rate
        if self.lm_head_frozen :
            if self.pretrained_model_name  == "distilbert-base-multilingual-cased" :
                lm_head = [self.translator.vocab_transform, self.translator.vocab_projector, self.translator.vocab_layer_norm]
            elif self.pretrained_model_name  == "bert-base-multilingual-uncased" or self.pretrained_model_name  == "bert-base-multilingual-cased" :
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
            elif self.pretrained_model_name  == "bert-base-multilingual-uncased" or self.pretrained_model_name  == "bert-base-multilingual-cased" :
                if hasattr(self.translator, "embeddings"):
                    model_embeddings = self.translator.embeddings
                elif hasattr(self.translator, "bert"):                    
                    model_embeddings = self.translator.bert.embeddings
                else:
                    import pdb;pdb.set_trace()
                    print("Error attr")
            elif self.pretrained_model_name  == "jhu-clsp/bibert-ende" :
                model_embeddings = self.translator.roberta.embeddings       
            elif self.pretrained_model_name  == "xlm-roberta-base" :
                model_embeddings = self.translator.roberta.embeddings                                            
            else:
                import pdb;pdb.set_trace()
                print ("Model name Error : args.pretrained_model_name") 
            for params in model_embeddings.parameters() :
                params.requires_grad=False    
        
        
        ########################################
        # the variable "task" does not exist

        self.ctc_beam_decoding = args.ctc_beam_decoding

        if args.ctc_beam_decoding:
            self.blank_idx = (
                tgt_dict.index(task.blank_symbol)
                if hasattr(tgt_dict, "blank_symbol")
                else tgt_dict.bos()
            )     

            labels=[]
            for i in range(len(tgt_dict)):
                labels.append(tgt_dict[i])

            self.beam_size = args.beam_size
            self.beam_decoder = CTCBeamDecoder(
                labels,
                model_path=args.kenlm_path,
                alpha=args.alpha,
                beta=args.beta,
                cutoff_top_n=10,
                cutoff_prob=1.0,
                beam_width=self.beam_size,
                num_processes=16,
                blank_id=self.blank_idx,
                log_probs_input=True
            )       
        
        self.voc_choosen = args.voc_choosen
        self.source_random_mask = False
        self.uniform_position_ids = False
        self.target_random_mask = False
        self.lm_random_mask = args.lm_random_mask
        self.lm_iter_num = args.lm_iter_num
        self.watch_lm_loss = args.watch_lm_loss
        self.lmk_loss_factor = 1
        if args.voc_choosen == 1:
            print("The target voc size is the loading form config")
            self.uniform_position_ids = False
        elif args.voc_choosen == 2:
            from transformers import PretrainedConfig, BertOnlyMLMHead
            output_projection_config = PretrainedConfig(hidden_size=args.encoder_embed_dim, hidden_act=args.activation_fn,
                                                        vocab_size=tgt_dict.__len__(), layer_norm_eps=1e-12)
            self.output_projection_layer = BertOnlyMLMHead(output_projection_config)
            self.output_projection_layer.apply(init_bert_params)
            
            print("The target voc size is use lenght of target.dict ")
            self.uniform_position_ids = False
            self.output_projection_warmup = args.output_projection_warmup
        elif args.voc_choosen == 3:
            self.uniform_position_ids = True   
        elif args.voc_choosen == 4:
            self.uniform_position_ids = True
            self.source_random_mask = True
            self.random_mask_rate = 0.15
        elif args.voc_choosen == 5:
            self.uniform_position_ids = True
            self.target_random_mask = True
            self.random_mask_rate = 0.15                          
        else:
            import pdb;pdb.set_trace()
            print("Error voc_choosen id")
            
        
        self.visualization = args.visualization
        if self.visualization :
            self.tgt_hidden_all=torch.tensor([])
            self.src_hidden_all=torch.tensor([])


        self.max_source_positions = args.max_source_positions
    
            
            


                                                      

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
        parser.add_argument(
            "--lmk-loss",
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
            help="COS or MSE or DIS",
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
        parser.add_argument(
            "--has-eos",        
            action="store_true",
            help="upsampling with eos , pervious version eos is padding's value",
        )            
        parser.add_argument(
            "--voc-choosen",
            type=int,
            default=1,
            help="1:use pretrain voc 2:use target dict size",
        )             
                   
        parser.add_argument(
            "--lm-random-mask",
            action="store_true",
            help="the input is masked before into lm model",
        )                          
       
        parser.add_argument(
            "--lm-iter-num",
            type=int,
            default=1,
            help="nunber of translation update",
        )                      
        
        parser.add_argument(
            "--watch-lm-loss",
            action="store_true",
            help="Watch lm and lmk losses with or without backpropagation",
        )       
        parser.add_argument(
            "--lm-mask-rate",
            type=float,
            default=0,
            help="input of lm modle is masked by the rate ",
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
                   
        if task.cfg.pretrained_model_path != "None" :
            translator_config = AutoConfig.from_pretrained(os.path.join(task.cfg.pretrained_model_path,"config.json"))
                   
        else:
            translator_config = AutoConfig.from_pretrained(args.pretrained_model_name)
        
        if not safe_hasattr(translator_config, "hidden_dropout_prob"):
            translator_config.dropout = args.dropout
        else:
            translator_config.hidden_dropout_prob = args.dropout
        
        if not safe_hasattr(translator_config, "hidden_dropout_prob"):
            translator_config.dropout = args.dropout
        else:
            translator_config.attention_probs_dropout_prob = args.dropout        
        
        if task.cfg.pretrained_model_path != "None" :
            translator = AutoModelForMaskedLM.from_pretrained(os.path.join(task.cfg.pretrained_model_path,"pytorch_model.bin"), config=translator_config)
        else:
            translator = AutoModelForMaskedLM.from_pretrained(args.pretrained_model_name, config=translator_config)
        
        if args.init_translator :
            translator.apply(init_bert_params)  
        


        if OmegaConf.is_config(args):
            OmegaConf.set_struct(args, True)

        # if task.cfg.no_atten_mask:
        vars(args)['no_atten_mask'] = task.cfg.no_atten_mask  
        vars(args)['debug'] = task.cfg.debug   
        vars(args)['visualization'] = task.cfg.visualization


        if task.cfg.ctc_beam_decoding:
            vars(args)['ctc_beam_decoding'] = task.cfg.ctc_beam_decoding
            vars(args)['beam_size'] = task.cfg.beam_size
            if task.cfg.kenlm_path != 'None':
                vars(args)['kenlm_path'] = task.cfg.kenlm_path
            else:
                vars(args)['kenlm_path'] = None
            vars(args)['alpha'] = task.cfg.alpha
            vars(args)['beta'] = task.cfg.beta 
                  
        return cls(args, translator, task.source_dictionary, task.target_dictionary )

               
    def forward(
        self, src_tokens, src_lengths, tgt_tokens, alignments, update_num,
        pretrained_lm=None, lm_loss_layer=-1, **kwargs
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
                logits, output_hidden_states, rate, src_upsample_tokens = self.translation(src_tokens, src_lengths, rate=self.num_upsampling_rate, **kwargs)
        else:
            logits, output_hidden_states, rate, src_upsample_tokens = self.translation(src_tokens, src_lengths, rate=self.num_upsampling_rate, **kwargs)
        
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
        
        if self.visualization :
            self.visualize(src_tokens,tgt_tokens,src_lengths)
        

        
        if self.ctc_beam_decoding:
            # logits, output_hidden_states, rate, src_tokens_upsample = self.translation(src_tokens, src_lengths, **kwargs) 

            ### jcx ###

            # logits, output_hidden_states, rate, src_tokens_upsample = self.translation(src_tokens, src_lengths, **kwargs) 
            logits, output_hidden_states, rate, src_tokens_upsample, attention_mask = self.translation(src_tokens, src_lengths, rate=self.num_upsampling_rate, **kwargs) 
            ### jcx ###
            if self.voc_choosen == 2:
                logits = self.output_projection_layer(output_hidden_states)            
            

            logits = F.log_softmax(logits, -1)

            # import pdb; pdb.set_trace()

            # from https://github.com/shawnkx/Fully-NAT/blob/781e11872c8f0dd8a6c1077c6d9dc160a7e472c4/fairseq-stable/fairseq/models/nat/cmlm_transformer.py
            topk = 10 # self.beam_size  # * 2
            decoder_topk_scores, decoder_topk_index = logits.topk(k=topk, dim=-1)

            # padding_mask = src_tokens_upsample[:, 1:].eq(self.pad)
            ### jcx ###
            padding_mask = ~attention_mask
            ### jcx ###

            # HACK: CTC beam-search requires the probability of blank, we put it in the end
            decoder_topk_scores = torch.cat([decoder_topk_scores, logits[..., self.blank_idx:self.blank_idx+1]], -1)
            decoder_topk_index = torch.cat([decoder_topk_index, decoder_topk_index.new_ones(*decoder_topk_index.size()[:-1], 1) * self.blank_idx], -1)
            if decoder_topk_index.size(0) > 1:
                decoder_topk_scores[..., 0].masked_fill_(padding_mask, 0.)
                decoder_topk_scores[..., -1].masked_fill_(padding_mask, 0.)
                decoder_topk_scores[..., 1:-1].masked_fill_(padding_mask.unsqueeze(-1), float("-Inf"))
                decoder_topk_index[...,0].masked_fill_(padding_mask, self.blank_idx)

            # for i in range(1):
            beam_results, beam_scores, timesteps, out_lens = self.beam_decoder.decode(decoder_topk_scores, decoder_topk_index)
            # unibeam_results, unibeam_scores, unitimesteps, uniout_lens = self.unibeam_decoder.decode(decoder_topk_scores, decoder_topk_index)

            # for i in range(len(beam_results)):
            #     if beam_scores[i][0] < unibeam_scores[i][0]:
            #         beam_results[i][0] = unibeam_results[i][0]
            #         out_lens[i][0] = uniout_lens[i][0]

            # import pdb; pdb.set_trace()
            
            _scores, _tokens = beam_scores[:,0].to(logits.device), beam_results[:, 0].to(logits.device).long()
            out_lens = out_lens.to(logits.device).type_as(_tokens)
            _scores = _scores[:, None].expand_as(_tokens)


            for i in range(src_tokens.size(0)):
                _tokens[i][out_lens[i][0]:] = self.pad

            ### jcx ###
            # _tokens = torch.ones_like(beam_results[:, 0]) * self.pad
            # for i in range(src_tokens.size(0)):
            #     max_beam = beam_scores[i].argmin()
            #     _tokens[i][:out_lens[i][max_beam]] = beam_results[i][max_beam][:out_lens[i][max_beam]]
            ### jcx ###

            # extra["padding_mask"] = new_arange(_tokens, *_tokens.size()) >= out_lens[:, :1]

            #########################################
        else:               
            logits, output_hidden_states, rate, src_upsample_tokens= self.translation(src_tokens, src_lengths, rate=self.num_upsampling_rate, **kwargs) 
            if self.voc_choosen == 2:
                logits = self.output_projection_layer(output_hidden_states)            

            _scores, _tokens = F.log_softmax(logits, dim=-1).max(-1)         

        # if self.debug:
        #     remove_duplicate_tokens = _tokens  
        # else :
        #     if _tokens.size(1) > 0 :
        #         unique_x, indices = torch.unique_consecutive(_tokens, return_inverse=True)
        #         indices -= indices.min(dim=1, keepdims=True)[0]
        #         remove_duplicate_tokens = torch.full_like(_tokens,self.pad)
        #         # remove_duplicate_score = torch.full_like(_scores,self.pad)
        #         # _scores  = remove_duplicate_score.scatter_(1, indices, _scores)
        #         remove_duplicate_tokens = remove_duplicate_tokens.scatter_(1, indices, _tokens)
        #     else:
        #         remove_duplicate_tokens = _tokens              



        
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

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)      

    def compute_lm_lsa_loss(self, output_rep, logits, src_tokens, tgt_tokens, tgt_output_rep, reduce=True):
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
    
    def compute_lmk_rep_loss(self, tgt_tokens, lm_token_embeddings, lm_bos_embeddings):
        with torch.no_grad():
            target_embeddings = torch.cat((lm_bos_embeddings, lm_token_embeddings), dim=1).type(self.translator.dtype)
        device = tgt_tokens.device
        bos = self.tgt_dict.bos() * torch.ones(tgt_tokens.shape[0], 1, dtype=torch.long, device=device)
        tgt_bos_tokens = torch.cat((bos, tgt_tokens), dim=1)      
        attention_mask=tgt_bos_tokens.ne(self.pad)  
        if self.no_atten_mask :
            output_translator = self.translator.forward(input_ids = tgt_bos_tokens, 
                                output_hidden_states=True, return_dict=True, 
                                    inputs_embeds=None)                 
        else:
            output_translator = self.translator.forward(input_ids = tgt_bos_tokens, 
                            attention_mask=attention_mask,  #encoder_attention_mask=attention_mask,
                            output_hidden_states=True, return_dict=True, 
                                inputs_embeds=None)      
        hidden_states = output_translator['hidden_states'][-1]    
        
          
        output_lmk_loss = 1 - F.cosine_similarity(hidden_states, target_embeddings, dim=2).mean()*self.lmk_loss_factor
        # output_lmk_loss = F.mse_loss(hidden_states,target_embeddings,reduction='mean').type(self.translator.dtype)*factor
        return output_lmk_loss
        
    def compute_lmk_loss(self, tgt_tokens, lm_token_embeddings,lm_loss_type,lm_lprobs=None, backpropagation=True, \
                         tgt_hidden_states=None, tgt_logits=None):
        def compute_loss( tgt_tokens, lm_token_embeddings,lm_loss_type,lm_lprobs,tgt_hidden_states,tgt_logits):    
            if tgt_hidden_states is None :
                device = tgt_tokens.device
                bos = self.tgt_dict.bos() * torch.ones(tgt_tokens.shape[0], 1, dtype=torch.long, device=device)
                tgt_bos_tokens = torch.cat((bos, tgt_tokens), dim=1)      
                attention_mask=tgt_bos_tokens.ne(self.pad)  
                if self.no_atten_mask :
                    output_translator = self.translator.forward(input_ids = tgt_bos_tokens, 
                                        output_hidden_states=True, return_dict=True, 
                                            inputs_embeds=None)                 
                else:
                    output_translator = self.translator.forward(input_ids = tgt_bos_tokens, 
                                    attention_mask=attention_mask,  #encoder_attention_mask=attention_mask,
                                    output_hidden_states=True, return_dict=True, 
                                        inputs_embeds=None)      
                hidden_states = output_translator['hidden_states'][-1][:, 1:, :]            
                logits = output_translator['logits'][:, 1:, :]
            else:
                hidden_states = tgt_hidden_states
                logits = tgt_logits
            
            if lm_loss_type == "DIS" :
                lprobs = self.get_normalized_probs([logits], log_probs=True)
                output_lmk_loss = F.kl_div(lprobs, lm_lprobs, reduction="batchmean", log_target=True) * 0.01
                
            elif lm_loss_type == "COS" :
                output_lmk_loss = (1 - F.cosine_similarity(hidden_states, lm_token_embeddings, dim=2).mean())
            elif lm_loss_type == "MSE" :
                output_lmk_loss = F.mse_loss(hidden_states,target_embeddings,reduction='mean').type(self.translator.dtype)
            else:
                import pdb;pdb.set_trace()
                raise NotImplementedError    
            return output_lmk_loss , hidden_states , logits
        
        if  backpropagation :
            loss = compute_loss( tgt_tokens, lm_token_embeddings,lm_loss_type,lm_lprobs, tgt_hidden_states, tgt_logits)
        else:
            with torch.no_grad():
                loss = compute_loss( tgt_tokens, lm_token_embeddings,lm_loss_type,lm_lprobs, tgt_hidden_states, tgt_logits) 
             
        return loss*self.lmk_loss_factor  
    
    def compute_lm_loss(self, output_rep, lprobs, src_tokens, lm_loss_type, tgt_tokens, \
                       lm_rep=None, lm_lprobs=None, backpropagation=True):
        def compute_loss(output_rep, lprobs, src_tokens, lm_loss_type, tgt_tokens, \
                            lm_rep, lm_lprobs):
            with torch.no_grad():
                bs, rep_seq_len ,_= output_rep.size()
                _, tgt_seq_len = tgt_tokens.size()
                target = tgt_tokens.repeat(1, rep_seq_len).view(bs, rep_seq_len, tgt_seq_len)
                bipart_no_pad = target.ne(self.pad)
                src_no_pad = src_tokens.ne(self.pad)
                bipart_lprobs = lprobs
                nll_loss = -bipart_lprobs.gather(dim=-1, index=target)#bs rep_seq_len tgt_seq_len
                nll_loss = nll_loss * bipart_no_pad
                match_index = nll_loss.argmin(1)
            
            if lm_loss_type == "DIS" :
                match_output_lprobs = bipart_lprobs[torch.arange(lprobs.shape[0]).unsqueeze(-1), match_index]
                output_lm_loss = F.kl_div(match_output_lprobs, lm_lprobs.to(match_output_lprobs.device),
                                            reduction="batchmean", log_target=True) * 0.01
            elif lm_loss_type == "COS" :
                match_output_rep = output_rep[torch.arange(output_rep.shape[0]).unsqueeze(-1), match_index]
                output_lm_loss = 1 - F.cosine_similarity(match_output_rep, lm_rep, dim=2).mean()
            elif lm_loss_type == "MSE" :
                match_output_rep = output_rep[torch.arange(output_rep.shape[0]).unsqueeze(-1), match_index]
                output_lm_loss = F.mse_loss(match_output_rep,lm_rep,reduction='mean')
            else:
                import pdb;pdb.set_trace()
                raise NotImplementedError
            return output_lm_loss
            
        if backpropagation :
            loss = compute_loss(output_rep, lprobs, src_tokens, lm_loss_type, tgt_tokens, lm_rep, lm_lprobs)
        else:
            with torch.no_grad():         
                loss = compute_loss(output_rep, lprobs, src_tokens, lm_loss_type, tgt_tokens, lm_rep, lm_lprobs)   
            
        return loss         

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
        attention_mask=src_tokens.ne(self.pad)  # paper not use the attenion_mask
        lm_outputs = model(src_tokens, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)     
        token_embeddings = lm_outputs['hidden_states'][output_layer]
        token_logits = lm_outputs['logits']
        # random_num = torch.rand(1)
        # token_embeddings = token_embeddings[-(int(random_num * self.use_drop_embedding)+1)]     
        # token_embeddings = token_embeddings[-1].detach()    
        bos_embeddings = token_embeddings[:, 0, :].detach().unsqueeze(1)
        token_embeddings = token_embeddings[:, 1:, :]
        token_logits = token_logits[:, 1:, :]
        src_tokens = src_tokens[:, 1:]    

        return token_embeddings, bos_embeddings, token_logits

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

        def integer_rate_mask(source, rate) :
            b,l = source.size()
            mask = source.ne(self.pad)  #ex : soruce=[7,7,7,pad] mask=[True True True False]
            mask_tokens = source.masked_fill(mask,self.mask)
            if rate == 7 :
                upsampled = torch.stack((source, mask_tokens, mask_tokens, mask_tokens, mask_tokens, mask_tokens, mask_tokens), dim=2).view(b, int(l*rate))               
            elif rate == 6 :
                upsampled = torch.stack((source, mask_tokens, mask_tokens, mask_tokens, mask_tokens, mask_tokens), dim=2).view(b, int(l*rate))               
            elif rate == 5 :
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
            return upsampled
             

        if self.upsample_fill_mask :
            if self.dynamic_upsampling :
                if self.dynamic_rate :
                    B, L = source.size(0), source.size(1)
                    new_length = torch.Tensor([L * rate]).int().item()    
                    if new_length > self.translator.config.max_position_embeddings :
                        rate= float(self.translator.config.max_position_embeddings)/float(L)
                
                if rate ==  int(rate) :
                    upsampled = integer_rate_mask(source, rate)
                    return upsampled, rate
                else :
                    insert_mask = True
                    t_x, t_mask, w, t_w, new_t_w, new_location = dynamic_upsample_token(source, insert_mask , rate, insertion_position=self.insert_position)  
                    return t_x, rate  
            else :
                upsampled = integer_rate_mask(source, rate)
                return upsampled, rate
                # b,l = source.size()
                # mask = source.ne(self.pad)  #ex : soruce=[7,7,7,pad] mask=[True True True False]
                # mask_tokens = source.masked_fill(mask,self.mask)
                # if rate == 5 :
                #     upsampled = torch.stack((source, mask_tokens, mask_tokens, mask_tokens, mask_tokens), dim=2).view(b, int(l*rate))                   
                # elif rate == 4 :
                #     upsampled = torch.stack((source, mask_tokens, mask_tokens, mask_tokens), dim=2).view(b, int(l*rate))                
                # elif rate == 3 :
                #     upsampled = torch.stack((source, mask_tokens, mask_tokens), dim=2).view(b, int(l*rate))
                # elif rate == 2:
                #     upsampled = torch.stack((source, mask_tokens), dim=2).view(b, int(l*rate))
                # else:
                #     print("Not support the rate in this setting")
                #     import pdb;pdb.set_trace()
        else:    
            if self.dynamic_upsampling :
                if self.dynamic_rate :
                    B, L = source.size(0), source.size(1)
                    new_length = torch.Tensor([L * rate]).int().item() 
                    if new_length > self.translator.config.max_position_embeddings :
                        rate= float(self.translator.config.max_position_embeddings)/float(L)  
                        
                if rate ==  int(rate) :  
                    upsampled =  torch.repeat_interleave(source, int(rate), dim=1)
                    return upsampled, rate   
                else :                            
                    insert_mask = False
                    t_x, t_mask, w, t_w, new_t_w, new_location = dynamic_upsample_token(source, insert_mask , rate) 
                    return t_x, rate 
            else:                        
                upsampled =  torch.repeat_interleave(source, int(rate), dim=1)
                return upsampled, rate

    def translation(self, src_tokens, src_lengths, upsampling_flag=True , rate=2, **kwargs):
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
            if upsampling_flag : 
                src_tokens_upsample, out_rate = self.upsampling(src_tokens, rate)
            else :
                src_tokens_upsample = src_tokens
                out_rate = 1
            src_tokens_upsample = torch.cat((bos, src_tokens_upsample), dim=1)  
            attention_mask=src_tokens_upsample.ne(self.pad)
            ###################
            # if ~self.has_eos and ~self.training and ~self.no_atten_mask :  # if eos is padding in the old version and generate for large batch 
            #     new_length = (src_lengths * rate).int() + 1  # +1 is for bos
            #     B, L = src_tokens_upsample.shape
            #     attention_mask = (torch.arange(0, L, dtype=torch.float32).unsqueeze(0).repeat(B, 1).to(src_tokens_upsample) < new_length.unsqueeze(-1))                
            ###################
        
            position_ids = self.calculate_position_ids(src_tokens_upsample)
            
            if self.no_atten_mask :
                output_translator = self.translator.forward(input_ids = src_tokens_upsample, 
                                    output_hidden_states=True, return_dict=True,position_ids=position_ids,
                                        inputs_embeds=None)                 
            else:

                output_translator = self.translator.forward(input_ids = src_tokens_upsample, 
                                attention_mask=attention_mask,  #encoder_attention_mask=attention_mask,
                                output_hidden_states=True, return_dict=True,position_ids=position_ids,
                                    inputs_embeds=None)

        
        if self.voc_choosen == 2:
            logits = None                     
        else:
            logits = output_translator['logits'][:,1:,:]    
        hidden_states = output_translator['hidden_states'][-1][:,1:,:]
        
        if self.ctc_beam_decoding:
            return logits, hidden_states, out_rate, src_tokens_upsample, attention_mask[:, 1:]        


        # ###################
        # import pdb;pdb.set_trace()
        # path='/livingrooms/valexsyu/dataset/model/mbert/pruned_models_BertForMaskedLM/pruned_V26458'
        # translator_config = AutoConfig.from_pretrained(os.path.join(path,"config.json"))
        # translator = AutoModelForMaskedLM.from_pretrained(os.path.join(path,"pytorch_model.bin"), config=translator_config)
        # output_ini = translator.forward(input_ids = src_tokens_upsample, 
        #                         attention_mask=attention_mask,  #encoder_attention_mask=attention_mask,
        #                         output_hidden_states=True, return_dict=True,position_ids=position_ids,
        #                             inputs_embeds=None)
        # logits_ini = output_ini['logits'][:,1:,:]    
        # hidden_states_ini = output_ini['hidden_states'][-1][:,1:,:]         
        
        # return logits, hidden_states, logits_ini, hidden_states_ini
        # ###################

        return logits, hidden_states, out_rate , src_tokens_upsample[:,1:]
    


    def calculate_position_ids(self, src_tokens_upsample):
        seq_length = src_tokens_upsample.shape[1]
        if isinstance(self.max_source_positions, tuple) :
            max_source_positions = self.max_source_positions[0]
        elif isinstance(self.max_source_positions, int) :
            max_source_positions = self.max_source_positions
        elif isinstance(self.max_source_positions, ListConfig):
            max_source_positions = self.max_source_positions[0]    
        else :
            import pdb;pdb.set_trace()
            print("self.max_source_positions is not defined the type")   
                 
        if self.uniform_position_ids and self.training == True :
            if seq_length <= max_source_positions :
                random_start_position = random.randint(1, max(1,max_source_positions-seq_length))
            else:
                random_start_position = 1
            position_ids = torch.arange(random_start_position,min(random_start_position+seq_length-1,max_source_positions)).to(src_tokens_upsample)
            bos_position = bos_position=torch.zeros(1,dtype=int).to(src_tokens_upsample)
            position_ids = torch.cat((bos_position,position_ids)).to(src_tokens_upsample).unsqueeze(0)
            
            # valex
            if seq_length > max_source_positions :
                expad_position_ids = torch.full((1,seq_length-max_source_positions), max_source_positions-1).to(position_ids)
                position_ids = torch.cat((position_ids,expad_position_ids), axis=1)
            # valex  
        else:
            random_start_position = 0
            position_ids = torch.arange(random_start_position,min(random_start_position+seq_length,max_source_positions)
                                        ).to(src_tokens_upsample).unsqueeze(0)
            if seq_length > max_source_positions :
                expad_position_ids = torch.full((1,seq_length-max_source_positions), max_source_positions-1).to(position_ids)
                position_ids = torch.cat((position_ids,expad_position_ids), axis=1)
            
            # if seq_length <= max_source_positions :
            #     random_start_position = random.randint(1, max(1,max_source_positions-seq_length))
            # else:
            #     random_start_position = 1
            # position_ids = torch.arange(random_start_position,min(random_start_position+seq_length-1,max_source_positions)).to(src_tokens_upsample)
            # bos_position = bos_position=torch.zeros(1,dtype=int).to(src_tokens_upsample)
            # position_ids = torch.cat((bos_position,position_ids)).to(src_tokens_upsample).unsqueeze(0)
            
            # # valex
            # if seq_length > max_source_positions :
            #     expad_position_ids = torch.full((1,seq_length-max_source_positions), max_source_positions-1).to(position_ids)
            #     position_ids = torch.cat((position_ids,expad_position_ids), axis=1)
            # # valex              
        return position_ids
    
    def _random_mask(self, target_tokens, rate):
        pad = self.tgt_dict.pad()
        bos = self.tgt_dict.bos()
        eos = self.tgt_dict.eos()
        mask = self.mask
        target_masks = (
            target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
        )
        target_score = target_tokens.clone().float().uniform_()
        target_score.masked_fill_(~target_masks, 2.0)
        target_length = target_masks.sum(1).float()
        target_length = target_length * target_length.clone().uniform_()*rate
        target_length = target_length + 1  # make sure to mask at least one token.

        _, target_rank = target_score.sort(1)
        target_cutoff = new_arange(target_rank) < target_length[:, None].long()
        prev_target_tokens = target_tokens.masked_fill(
            target_cutoff.scatter(1, target_rank, target_cutoff), mask
        )
        return prev_target_tokens    
    
    def visualize(self, src_tokens, tgt_tokens, src_lengths):
        # device = tgt_tokens.device
        # bos = self.tgt_dict.bos() * torch.ones(tgt_tokens.shape[0], 1, dtype=torch.long, device=device)
        # tgt_bos_tokens = torch.cat((bos, tgt_tokens), dim=1)      
        # attention_mask=tgt_bos_tokens.ne(self.pad)  
        # if self.no_atten_mask :
        #     output_translator = self.translator.forward(input_ids = tgt_bos_tokens, 
        #                         output_hidden_states=True, return_dict=True, 
        #                             inputs_embeds=None)                 
        # else:
        #     output_translator = self.translator.forward(input_ids = tgt_bos_tokens, 
        #                     attention_mask=attention_mask,  #encoder_attention_mask=attention_mask,
        #                     output_hidden_states=True, return_dict=True, 
        #                         inputs_embeds=None)     
             
        # tgt_hidden_states = output_translator['hidden_states'][-1]  
        
        pad_mask = (tgt_tokens != self.tgt_dict.pad()) & (
                    tgt_tokens != self.tgt_dict.eos()) & (
                    tgt_tokens != self.tgt_dict.bos())    
        target_lengths = pad_mask.sum(-1)    
        import pdb;pdb.set_trace() ##
        tgt_logits, tgt_hidden_states, tgt_logits_ini, tgt_hidden_states_ini= self.translation(tgt_tokens, target_lengths, upsampling_flag=False,rate=self.num_upsampling_rate) 
        src_logits, src_hidden_states, src_logits_ini, src_hidden_states_ini= self.translation(src_tokens, src_lengths,rate=self.num_upsampling_rate) 
        tgt_translate_token = F.softmax(tgt_logits, dim=-1).argmax(-1)
        src_translate_token = F.softmax(src_logits, dim=-1).argmax(-1)
        tgt_translate_token_ini = F.softmax(tgt_logits_ini, dim=-1).argmax(-1)
        src_translate_token_ini = F.softmax(src_logits_ini, dim=-1).argmax(-1)        
        self.tgt_hidden_all=torch.cat((self.tgt_hidden_all,tgt_hidden_states.reshape(-1,768).cpu()),0)
        self.src_hidden_all=torch.cat((self.src_hidden_all,src_hidden_states.reshape(-1,768).cpu()),0)
        tgt_len=self.tgt_hidden_all.size(0)
        import numpy as np
        

        
        
        all_hidden=torch.cat((self.tgt_hidden_all,self.src_hidden_all),0).cpu()
        normalized_tensor = torch.nn.functional.normalize(all_hidden, dim=1)
        print(normalized_tensor.size())
        # Convert embeddings to numpy array
        normalized_tensor_np = normalized_tensor.numpy()        
        tsne = TSNE(n_components=3, perplexity=30, random_state=42)
        tsne_embeddings = tsne.fit_transform(normalized_tensor_np)
        
        tsne = TSNE(n_components=3, perplexity=50, random_state=42)
        tsne_embeddings_1 = tsne.fit_transform(normalized_tensor_np)        

        # pca = PCA(n_components=2)
        # pca_embeddings = pca.fit_transform(normalized_tensor_np)
        
        print("TSEN translation done")
        tgt_len=self.tgt_hidden_all.size(0)
        fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1) # two axes on figure
        # plt.figure(figsize=(10, 8))
        ax1.scatter(tsne_embeddings[:tgt_len, 0], tsne_embeddings[:tgt_len, 1], tsne_embeddings[:tgt_len, 2], c='whitesmoke')
        ax1.scatter(tsne_embeddings[tgt_len:, 0], tsne_embeddings[tgt_len:, 1], tsne_embeddings[tgt_len:, 2], c='oldlace')        
        ax1.scatter(tsne_embeddings[:20, 0], tsne_embeddings[:20, 1], tsne_embeddings[:20, 2], c='red', marker='x')
        ax1.scatter(tsne_embeddings[tgt_len:tgt_len+20, 0], tsne_embeddings[tgt_len:tgt_len+20, 1], tsne_embeddings[tgt_len:tgt_len+20, 2], c='blue', marker='x')
        
        ax2.scatter(tsne_embeddings_1[:tgt_len, 0], tsne_embeddings_1[:tgt_len, 1], tsne_embeddings_1[:tgt_len, 2], c='whitesmoke')
        ax2.scatter(tsne_embeddings_1[tgt_len:, 0], tsne_embeddings_1[tgt_len:, 1], tsne_embeddings_1[tgt_len:, 2], c='oldlace')        
        ax2.scatter(tsne_embeddings_1[:20, 0], tsne_embeddings_1[:20, 1], tsne_embeddings_1[:20, 2], c='red', marker='x')
        ax2.scatter(tsne_embeddings_1[tgt_len:tgt_len+20, 0], tsne_embeddings_1[tgt_len:tgt_len+20, 1], tsne_embeddings_1[tgt_len:tgt_len+20, 2], c='blue', marker='x')            
        plt.savefig('checkpoints/m-8-3-3-K12-UF20M-test/plot_tsne' + str(tgt_len) + '.png', dpi=300, bbox_inches='tight')
        
                
             

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
    args.lmk_loss  = safe_getattr( args, "lmk_loss", False )
    # args.lm_loss_dis  = safe_getattr( args, "lm_loss_dis", False )
    args.lm_head_frozen  = safe_getattr( args, "lm_head_frozen", False )
    args.embedding_frozen  = safe_getattr( args, "embedding_frozen", False )
    args.lm_loss_self  = safe_getattr( args, "lm_loss_self", False )
    args.lm_loss_type  = safe_getattr( args, "lm_loss_type", "COS" )
    # args.lm_loss_layer  = safe_getattr( args, "lm_loss_layer", -1 )
    args.lm_tr_layer  = safe_getattr( args, "lm_tr_layer", [-1] )
    args.lm_st_layer  = safe_getattr( args, "lm_st_layer", [-1] )
    args.lm_start_step = safe_getattr( args, "lm_start_step", 75000)
    args.output_projection_warmup = safe_getattr( args, "output_projection_warmup", 7000)
    
    
    args.upsample_fill_mask  = safe_getattr( args, "upsample_fill_mask", False )
    args.dynamic_upsampling  = safe_getattr( args, "dynamic_upsampling", False )
    args.dynamic_rate = safe_getattr( args, "dynamic_rate", False )
    args.insert_position  = safe_getattr( args, "insert_position", "uniform" )
    
    args.no_atten_mask = safe_getattr(args, "no_atten_mask", False )
    args.debug = safe_getattr(args, "debug", False )
    args.has_eos = safe_getattr(args, "has_eos", False )
    args.visualization = safe_getattr(args, "visualization", False )
    args.lm_random_mask = safe_getattr(args, "lm_random_mask", False )
    args.lm_iter_num = safe_getattr(args, "lm_iter_num", 1 )
    args.watch_lm_loss = safe_getattr(args, "watch_lm_loss", False)
    args.lm_mask_rate = safe_getattr(args, "lm_mask_rate", 0)
    args.voc_choosen = safe_getattr(args, "voc_choosen", 1)
    


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
    
    
    # ctc beam decoding config
    args.ctc_beam_decoding = safe_getattr(args, "ctc_beam_decoding", False )
    args.beam_size = safe_getattr(args, "beam_size", 1 )
    args.kenlm_path = safe_getattr(args, "kenlm_path", None )
    args.alpha = safe_getattr(args, "alpha", 0.0 )
    args.beta = safe_getattr(args, "beta", 0.0 )    
    
    
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




# @register_model("nat_pretrained_lm_lsa_model") 
# class NATPretrainedLMLsaModel(NATPretrainedModel):
#     def __init__(self, args, translator, src_dict, tgt_dict):
#         super().__init__(args, translator, src_dict, tgt_dict)    
#         if args.ctc_beam_decoding:
#             self.blank_idx = (
#                 tgt_dict.index(task.blank_symbol)
#                 if hasattr(tgt_dict, "blank_symbol")
#                 else tgt_dict.bos()
#             )         
    
#     def compute_lm_loss(self, output_rep, lprobs, src_tokens, lm_loss_type, tgt_tokens, \
#                        lm_rep=None, lm_lprobs=None, backpropagation=True):
        
#         def compute_lm_lsa_loss(self, output_rep, logits, src_tokens, tgt_tokens, tgt_output_rep, reduce=True):
#             with torch.no_grad():    
#                 bs, rep_seq_len ,_= output_rep.size()
#                 _, tgt_seq_len = tgt_tokens.size()
#                 target = tgt_tokens.repeat(1, rep_seq_len).view(bs, rep_seq_len, tgt_seq_len)
#                 bipart_no_pad = target.ne(self.pad)
#                 src_no_pad = src_tokens.ne(self.pad)
#                 bipart_lprobs = lprobs
#                 nll_loss = -bipart_lprobs.gather(dim=-1, index=target)#bs rep_seq_len tgt_seq_len
#                 nll_loss = nll_loss * bipart_no_pad

#                 nll_loss_numpy = nll_loss.detach()
#                 tgt_output_rep = tgt_output_rep.detach()
#             lm_loss = torch.zeros(1).to(src_tokens.device)
#             for batch_id in range(bs):
#                 no_pad_num = bipart_no_pad[batch_id, 0].sum()
#                 src_no_pad_num = src_no_pad[batch_id].sum()
#                 output_tokens = logits[batch_id].argmax(-1)
#                 output_tokens_blank_mask = output_tokens.eq(self.src_dict.bos()).view(-1,1).repeat(1,tgt_seq_len)
#                 nll_loss_numpy_line = nll_loss_numpy[batch_id]
#                 nll_loss_numpy_line = nll_loss_numpy_line.masked_fill_(output_tokens_blank_mask, float(10^8))
#                 raw_index, col_index = lsa(nll_loss_numpy_line[:src_no_pad_num, :no_pad_num].cpu().numpy())
#                 lm_loss = ((1 - F.cosine_similarity(output_rep[batch_id][raw_index], tgt_output_rep[batch_id][col_index])).mean())+ lm_loss
                
#             return lm_loss/bs
                     
#         def compute_loss(output_rep, lprobs, src_tokens, lm_loss_type, tgt_tokens, \
#                             lm_rep, lm_lprobs):
#             with torch.no_grad():
#                 bs, rep_seq_len ,_= output_rep.size()
#                 _, tgt_seq_len = tgt_tokens.size()
#                 target = tgt_tokens.repeat(1, rep_seq_len).view(bs, rep_seq_len, tgt_seq_len)
#                 bipart_no_pad = target.ne(self.pad)
#                 src_no_pad = src_tokens.ne(self.pad)
#                 bipart_lprobs = lprobs
#                 nll_loss = -bipart_lprobs.gather(dim=-1, index=target)#bs rep_seq_len tgt_seq_len
#                 nll_loss = nll_loss * bipart_no_pad
#                 match_index = nll_loss.argmin(1)
            
#             if lm_loss_type == "DIS" :
#                 match_output_lprobs = bipart_lprobs[torch.arange(lprobs.shape[0]).unsqueeze(-1), match_index]
#                 output_lm_loss = F.kl_div(match_output_lprobs, lm_lprobs.to(match_output_lprobs.device),
#                                             reduction="batchmean", log_target=True) * 0.01
#             elif lm_loss_type == "COS" :
#                 match_output_rep = output_rep[torch.arange(output_rep.shape[0]).unsqueeze(-1), match_index]
#                 output_lm_loss = 1 - F.cosine_similarity(match_output_rep, lm_rep, dim=2).mean()
#             elif lm_loss_type == "MSE" :
#                 match_output_rep = output_rep[torch.arange(output_rep.shape[0]).unsqueeze(-1), match_index]
#                 output_lm_loss = F.mse_loss(match_output_rep,lm_rep,reduction='mean')
#             else:
#                 import pdb;pdb.set_trace()
#                 raise NotImplementedError
#             return output_lm_loss
            
#         if backpropagation :
#             loss = compute_lm_lsa_loss(output_rep, lprobs, src_tokens, lm_loss_type, tgt_tokens, lm_rep, lm_lprobs)
#         else:
#             with torch.no_grad():         
#                 loss = compute_lm_lsa_loss(output_rep, lprobs, src_tokens, lm_loss_type, tgt_tokens, lm_rep, lm_lprobs)   
            
#         return loss          