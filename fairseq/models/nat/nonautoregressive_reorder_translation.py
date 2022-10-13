
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
    BaseFairseqModel
)
from fairseq.models.nat.nonautoregressive_encoder import (
    NATRobertaModel,
    NATRobertaEcoder
)

from fairseq.utils import safe_getattr, safe_hasattr
logger = logging.getLogger(__name__)





@register_model("nonautoregressive_reorder_translation") 
class NATReorderTranslation(BaseFairseqModel):
    def __init__(self, args, reorder, translator, src_dict, tgt_dict):
        super().__init__() 
        self.reorder_translation = args.reorder_translation
        self.reorder = reorder
        self.translator = translator
        self.num_upsampling_rate = args.num_upsampling_rate
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.bos = tgt_dict.bos()
        self.eos = tgt_dict.eos()
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()              
        #self.freeze_module = args.freeze_module
        self.pretrained_reorder = args.pretrained_reorder
        self.pretrained_translation = args.pretrained_translation
        self.voc_token = torch.range(0, src_dict.__len__()-1,dtype=float,requires_grad=True)
        self.voc_size = src_dict.__len__()
        self.kl_div = nn.KLDivLoss(reduction="batchmean",log_target=True)
        self.mse = nn.MSELoss(reduce=False)
        self.reorder_factor = 2
        self.TEST_STEP=0 
        self.TEST_NUM=1000   

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        NATRobertaModel.add_args(parser)
              
        parser.add_argument(
            "--reorder-translation",
            choices=['reorder_translation', 'reorder', 'translation'],
            default="reorder-translation",
            help="choise the model type, reorder-translation/reorder/translation",
        )    

        parser.add_argument(
            "--num-upsampling-rate",
            type=int,
            default=1,
            help="The multiplier value of the source upsampling",
        )        
 
        parser.add_argument(
            "--pretrained-reorder",
            type=str,
            default=None,
            help="path of pretrained reorder path",
        )         
        parser.add_argument(
            "--pretrained-translation",
            type=str,
            default=None,
            help="path of pretrained translation path",
        )           
        parser.add_argument(
            "--freeze-module",
            choices=["reorder", "translator","None"],
            default="None",
            help="choose a module to freeze",
        )  
        """
        parser.add_argument(
            "--global-token",
            action="store_true",
            help="if set, use global_token but not calculate loss in nat_ctc_loss",
        )        
        """ 
    def load_pretrained_model(self,):
        def load_checkpoint(model, path, num_rm_keystr=0):
            checkpoint = torch.load(path)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model'].items():
                if k[:num_rm_keystr] == 'translat':
                    pass
                else:
                    name = k[num_rm_keystr:] # remove `translator.`
                    new_state_dict[name] = v  
            # load params
            model.load_state_dict(new_state_dict) 
        if self.pretrained_reorder is not None:
            load_checkpoint(self.reorder, self.pretrained_reorder, 8)
            print("Load pretrained reorder , path:{}".format(self.pretrained_reorder))
        if self.pretrained_translation is not None:
            load_checkpoint(self.translator, self.pretrained_translation, 11) 
            print("Load pretrained translator , path:{}".format(self.pretrained_translation))

               
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
            
        if args.reorder_translation == "reorder_translation" :
            #-----Force setting the reorder -----#
            # global_token = False
            # encoder_causal_atten = False
            # max_positions = 512
            # max_source_positions = 512
            temp_encoder_causal_attn = args.encoder_causal_attn             
            temp_global_token = args.global_token            
            args.max_positions = 512     
            args.max_source_positions = 512             
            args.encoder_causal_attn = False
            args.global_token = False

            reorder_encoder = NATRobertaEcoder(args, task.source_dictionary, task.source_dictionary)
            args.global_token = temp_global_token
            args.encoder_causal_attn = temp_encoder_causal_attn
            reorder = NATRobertaModel(args, reorder_encoder, task.source_dictionary ,  task.source_dictionary)
            
            args.max_positions = 1024                                       
            args.max_source_positions = 1024    
            translator_encoder = NATRobertaEcoder(args, task.source_dictionary, task.target_dictionary)      
            translator = NATRobertaModel(args,  translator_encoder, task.source_dictionary, task.target_dictionary)  
                        
        elif args.reorder_translation == "reorder" : 
            reorder_encoder = NATRobertaEcoder(args, task.source_dictionary, task.source_dictionary)
            reorder = NATRobertaModel(args, reorder_encoder, task.source_dictionary ,  task.source_dictionary)
            translator = None
            
        elif args.reorder_translation == "translation" :
            translator_encoder = NATRobertaEcoder(args, task.source_dictionary, task.target_dictionary)   
            reorder = None    
            translator = NATRobertaModel(args, translator_encoder, task.source_dictionary, task.target_dictionary)  
           
        else:
            import pdb;pdb.set_trace()
            print("Error: wrong input in args.reorder_translation")       
        
        if OmegaConf.is_config(args):
            OmegaConf.set_struct(args, True)

        
        return cls(args, reorder, translator, task.source_dictionary, task.target_dictionary)

   
    
    def forward(
        self, src_tokens, src_lengths, src_noise_tokens, 
        tgt_tokens, freeze_module,  **kwargs):
        if self.reorder_translation == "reorder_translation" :
            output = self.reorder_translation_forward(src_tokens, src_lengths, tgt_tokens, freeze_module, **kwargs )
        elif self.reorder_translation == "reorder" : 
            output = self.reorder_forward(src_tokens, src_lengths, tgt_tokens, **kwargs)
        elif self.reorder_translation == "translation" : 
            output = self.translator_forward(src_noise_tokens, src_lengths, tgt_tokens, **kwargs)
        
        return output
            
                
    def reorder_forward(
        self, src_tokens, src_lengths, tgt_tokens, **kwargs
    ):
        outputs = self.reorder(src_tokens=src_tokens, src_lengths=None, 
                               tgt_tokens=tgt_tokens, 
                               **kwargs)
        
        vocabulary_mask = torch.ones((outputs['word_ins']['out'].size(0),
                                      outputs['word_ins']['out'].size(2)),
                                      dtype=bool,
                                      device=src_tokens.device).scatter_(1,src_tokens,0).unsqueeze(1)
        
        outputs['word_ins']['out'] = outputs['word_ins']['out'].masked_fill(vocabulary_mask, float("-inf"))        
        outputs['word_ins']['loss_type'] = "NLL"
        return outputs
        
        
    def translator_forward(
        self, src_tokens, src_lengths, tgt_tokens, token_embeddings=None, **kwargs
    ): 
        
        upsampled_toks = self.upsampling(src_tokens, self.num_upsampling_rate)
        if token_embeddings is not None:
            upsampled_token_embedding = self.upsampling(token_embeddings, self.num_upsampling_rate)
        else:
            upsampled_token_embedding = None
        outputs = self.translator(src_tokens=upsampled_toks, src_lengths=None, 
                                  tgt_tokens=tgt_tokens, token_embeddings=upsampled_token_embedding, **kwargs)       
        return outputs
    
    def reorder_translation_forward(
        self, src_tokens, src_lengths,tgt_tokens, freeze_module=None, **kwargs
    ):
        def kl_div(reorder_onehot , src_tokens):
            q = F.log_softmax(torch.sum(reorder_onehot, dim=1))
            src_onehot = F.one_hot(src_tokens, num_classes=self.voc_size).type_as(reorder_onehot)
            p = F.log_softmax(torch.sum(src_onehot, dim=1))
            kl_loss = self.kl_div(q , p)
            return kl_loss            
        
        def mse_weight(reorder_onehot , src_tokens, src_len, miss_len, factor=1):
            q = torch.sum(reorder_onehot, dim=1)
            src_onehot = F.one_hot(src_tokens, num_classes=self.voc_size).type_as(reorder_onehot)
            p = torch.sum(src_onehot, dim=1)
            weight = torch.div(torch.FloatTensor(miss_len).to(src_len.device), src_len)
            mse_loss = torch.mul(self.mse(q, p).sum(-1), weight).mean()
            #mse_loss = self.mse(q, p, reduction='none')
            return mse_loss
        def miss_token_sentance(prediction, target) :
            miss_len=[]
            miss_tokens=[]
            for index, (pred, tgt) in enumerate(zip(prediction, target)):
                miss_tok = ([x for x in tgt.tolist() if x not in pred.tolist()])
                miss_len.append(len(miss_tok))
                miss_tokens.append(miss_tok)
            return miss_len, miss_tokens



        no_grad_condition = True if freeze_module=="reorder" and self.training else False
        with torch.set_grad_enabled(not no_grad_condition):            
            reorder_logit = self.reorder_forward(src_tokens, src_lengths, None, **kwargs)['word_ins']['out']
            
            reorder_onehot = F.gumbel_softmax(reorder_logit,tau=1, hard=True)   
            embed_weight = self.translator.encoder.sentence_encoder.embed_tokens._parameters['weight']
            token_embeddings = torch.matmul(reorder_onehot,embed_weight)

            b,l = src_tokens.size()
            mask_pad = src_tokens.eq(self.pad).unsqueeze(-1)
            token_embeddings = token_embeddings.masked_scatter(mask_pad, embed_weight[self.pad].expand(b,l,-1))
            """
            for i in range(src_tokens.size(0)) :
                for j in range(src_tokens.size(1)):
                    if src_tokens[i][j] == self.pad :
                        token_embeddings[i][j] = embed_weight[self.pad]
            """
            

            self.voc_token = self.voc_token.type_as(reorder_logit)  
            reorder_voc = torch.matmul(reorder_onehot, self.voc_token.unsqueeze(-1)).type_as(src_tokens).squeeze(-1)
            reorder_voc = self.padding_from_source(reorder_voc, self.pad, src_tokens)
            if not freeze_module=="reorder" :
                #kl_loss = kl_div(reorder_onehot , src_tokens)*self.reorder_factor
                miss_len, miss_tok = miss_token_sentance(reorder_voc, src_tokens)
                mse_loss = mse_weight(reorder_onehot , src_tokens, src_lengths, miss_len, self.reorder_factor)
        no_grad_condition = True if freeze_module=="translator" and self.training else False
        with torch.set_grad_enabled(not no_grad_condition): 
            outputs = self.translator_forward(src_tokens=reorder_voc, src_lengths=None, 
                                      tgt_tokens=tgt_tokens,token_embeddings=token_embeddings, **kwargs)        

        if no_grad_condition :
            outputs['word_ins']['out'].requires_grad_(True)  
        
        if not freeze_module=="reorder" :
            outputs["reorder"] = {
                "loss": mse_loss,
                "factor" : self.reorder_factor,
                "loss_type": "loss",
            }


        #test======================================
        if self.TEST_STEP % self.TEST_NUM == 0 :
            print("Step:{}".format(self.TEST_STEP))
            print("Sourcee:{}".format(src_tokens[0]))
            print("Reorder:{}".format(reorder_voc[0]))

            test_miss_len, test_miss_token = miss_token_sentance([reorder_voc[0]], [src_tokens[0]])
            #test_miss_token = ([x for x in src_tokens[0].tolist() if x not in reorder_voc[0].tolist()])
            test_kl_loss = kl_div(torch.unsqueeze(reorder_onehot[0],dim=0) , 
                                     torch.unsqueeze(src_tokens[0],dim=0))

            test_mse_loss = mse_weight(torch.unsqueeze(reorder_onehot[0],dim=0) , 
                                     torch.unsqueeze(src_tokens[0],dim=0),
                                     src_lengths[0], test_miss_len, self.reorder_factor )
            print("Reorder_Miss_token:{}    len:{}     KL_DISS:{:2.5f}   MSE:{:2.5f}".format(
                                     test_miss_token,test_miss_len,test_kl_loss,test_mse_loss))
            test_output = torch.argmax(outputs['word_ins']['out'][0], dim=-1)
            print("Pred:{}".format(torch.unique_consecutive(test_output[test_output!= 10160])))     
            print("tgtt:{}".format(tgt_tokens[0]))        
            
        self.TEST_STEP += 1
        #=========================================test

        return outputs    
        
    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)
      
        
    def initialize_output_tokens(self, src_tokens):
        initial_output_tokens = src_tokens
        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        )
        
        return DataOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )        
    def forward_inference(self, data_out, **kwargs):
        step = data_out.step
        output_tokens = data_out.output_tokens
        output_scores = data_out.output_scores
        history = data_out.history        
        
        if self.reorder_translation == "reorder_translation" :
            reorder_logit = self.reorder_forward(src_tokens=output_tokens, src_lengths=None, 
                                         tgt_tokens=None, **kwargs)['word_ins']['out']    
                                 
            reorder_voc = torch.argmax(reorder_logit, dim=-1) 
            
            reorder_pad = self.padding_from_source(reorder_voc, self.pad, output_tokens)
            for index,(source, reorder) in enumerate(zip(output_tokens, reorder_pad)):
                miss_token = ([x for x in source if x not in reorder])
                if len(miss_token) >3 :
                    print("sourcee:{}".format(source))                 
                    print("reorder:{}".format(reorder))  
                    print("paddddd:{}".format(reorder_pad[index]))
                    print("MISS TOKEN:{}".format(miss_token))
                    print("=======================================================================")   
                                   
                    
                
            
            reorder_voc = self.padding_from_source(reorder_voc, self.pad, output_tokens)
            logit = self.translator_forward(src_tokens=reorder_voc, src_lengths=None, 
                                      tgt_tokens=None, **kwargs)['word_ins']['out']               
                           
        elif self.reorder_translation == "reorder" : 
            logit = self.reorder_forward(src_tokens=output_tokens, src_lengths=None, 
                                         tgt_tokens=None, **kwargs)['word_ins']['out']

        elif self.reorder_translation == "translation" : 
            logit = self.translator_forward(src_tokens=output_tokens, src_lengths=None,                                           
                                         tgt_tokens=None, **kwargs)['word_ins']['out']
            
        _scores, _tokens = F.log_softmax(logit,-1).max(-1)   
        output_scores=output_scores.type_as(_scores)
        output_tokens = _tokens
        output_scores = _scores 
        #output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        #output_scores.masked_scatter_(output_masks, _scores[output_masks]) 
        if history is not None:
            history.append(output_tokens.clone())                
        return data_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )
        
    def upsampling(self, source_toks, num_upsampling_rate): 
        upsampled_toks =  torch.repeat_interleave(source_toks, num_upsampling_rate, dim=1)
        return upsampled_toks

    def padding_from_source(self, tgt, pad_id, src):
        src_masks  = src.eq(pad_id)             
        tgt = tgt.masked_fill(src_masks,value=pad_id)
        return tgt        
       


        
        
@register_model_architecture(
    "nonautoregressive_reorder_translation", "nonautoregressive_reorder_translation"
)
def base_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 6)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 12)

    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = safe_getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = safe_getattr(args, "pooler_dropout", 0.0)

    args.max_positions = safe_getattr(args, "max_positions", 1024)
    args.max_source_positions = safe_getattr(args, "max_source_positions", 1024)
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
    
    # NAT config
    args.encoder_causal_attn = safe_getattr(args, "encoder_causal_attn", False)
    
    # Reorder_Translation config
    args.reorder_translation = safe_getattr(args, "reorder_translation", "reorder_translation")
    args.freeze_module = safe_getattr(args, "freeze_module", "None")
    args.pretrained_reorder = safe_getattr(args, "pretrained_reorder", None )
    args.pretrained_translation = safe_getattr(args, "pretrained_translation", None )
    args.num_upsampling_rate = safe_getattr(args, "num_upsampling_rate", 1 )
    #args.num_upsampling_rate = safe_getattr(args, "global_token", False )


        