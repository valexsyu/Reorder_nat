
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
            args.max_positions = 512     
            args.max_source_positions = 512             
              
            if args.encoder_causal_attn :
                args.encoder_causal_attn = False
                reorder_encoder = NATRobertaEcoder(args, task.source_dictionary, task.source_dictionary)
                args.encoder_causal_attn = True  
                reorder = NATRobertaModel(args, reorder_encoder, task.source_dictionary ,  task.source_dictionary)
            else:
                reorder_encoder = NATRobertaEcoder(args, task.source_dictionary, task.source_dictionary) 
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
        self, src_tokens, src_lengths, prev_output_tokens, 
        tgt_tokens, **kwargs):
        
        if self.reorder_translation == "reorder_translation" :
            output = self.reorder_translation_forward(src_tokens, src_lengths, prev_output_tokens, 
                                                      tgt_tokens, **kwargs )
        elif self.reorder_translation == "reorder" : 
            output = self.reorder_forward(src_tokens, src_lengths, prev_output_tokens, 
                                          tgt_tokens, **kwargs)
        elif self.reorder_translation == "translation" : 
            output = self.translator_forward(src_tokens, src_lengths, prev_output_tokens, 
                                             tgt_tokens, **kwargs)
        return output
            
        
            
                
    def reorder_forward(
        self, src_tokens, src_lengths, prev_output_tokens,                                                       
        tgt_tokens, **kwargs
    ):
        
        outputs = self.reorder(src_tokens=src_tokens, src_lengths=None, 
                               prev_output_tokens=src_tokens, tgt_tokens=tgt_tokens, 
                               **kwargs)
        
        vocabulary_mask = torch.ones(outputs['word_ins']['out'].size(0),outputs['word_ins']['out'].size(2)).index_fill_(1,src_tokens,0)
        outputs['word_ins']['out'] = outputs['word_ins']['out'].masked_fill(vocabulary_mask, float("-inf"))
              
        
        return outputs
        
        
    def translator_forward(
        self, src_tokens, src_lengths, prev_output_tokens,                                                       
        tgt_tokens, **kwargs
    ):
        upsampled_toks = self.upsampling(src_tokens)
        outputs = self.translator(src_tokens=src_tokens, src_lengths=None, 
                               prev_output_tokens=upsampled_toks, tgt_tokens=tgt_tokens, 
                               **kwargs)

        return outputs
    
    def reorder_translation_forward(
        self, src_tokens, src_lengths, prev_output_tokens,                                                       
        tgt_tokens, **kwargs
    ):
        reorder_outputs = self.reorder(src_tokens=src_tokens, src_lengths=None, 
                               prev_output_tokens=src_tokens, tgt_tokens=tgt_tokens, 
                               **kwargs)
        
        vocabulary_mask = torch.ones((reorder_outputs['word_ins']['out'].size(0),
                                      reorder_outputs['word_ins']['out'].size(2)),
                                      dtype=bool,
                                      device=src_tokens.device).scatter_(1,src_tokens,0).unsqueeze(1)
        
        reorder_outputs['word_ins']['out'] = reorder_outputs['word_ins']['out'].masked_fill(vocabulary_mask, float("-inf"))
        
        reorder_onehot = F.gumbel_softmax(reorder_outputs['word_ins']['out'], hard=True).type_as(src_tokens)
        reorder_voc = torch.argmax(reorder_onehot, dim=-1)
        
        upsampled_toks = self.upsampling(reorder_voc)
        outputs = self.translator(src_tokens=src_tokens, src_lengths=None, 
                               prev_output_tokens=upsampled_toks, tgt_tokens=tgt_tokens, 
                               **kwargs)        
        

        return outputs    
        
    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)
      
        
    def initialize_output_tokens(self, src_tokens, num_upsampling_rate=3):


        initial_output_tokens = torch.repeat_interleave(src_tokens, num_upsampling_rate, dim=1)
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
        
        output_masks  = output_tokens.ne(self.pad)
        logit, inner_state = self.encoder(src_tokens = output_tokens, features_only=False, 
                                          return_all_hiddens=False, classification_head_name=None, 
                                          causal_attn=self.causal_attn,  **kwargs)   
        _scores, _tokens = F.log_softmax(logit,-1).max(-1)   
        output_scores=output_scores.type_as(_scores)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks]) 
        if history is not None:
            history.append(output_tokens.clone())                
        return data_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )
        
    def upsampling(self, source_toks): #valex
        upsampled_toks =  torch.repeat_interleave(source_toks, self.num_upsampling_rate, dim=1)
        return upsampled_toks   
        
        
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


        