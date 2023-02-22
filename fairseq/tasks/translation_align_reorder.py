# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import imp
import torch
import json
from argparse import Namespace
from fairseq import metrics,utils
import numpy as np
import logging
from fairseq.data import LanguagePairDataset, encoders, Dictionary
from fairseq.dataclass import ChoiceEnum
from fairseq.tasks import register_task
from fairseq.tasks.translation import (
    TranslationConfig,
    TranslationTask,
    load_langpair_dataset,
)
from fairseq.utils import new_arange
from typing import Optional
from transformers import AutoModel, AutoModelForMaskedLM

NOISE_CHOICES = ChoiceEnum(["random_delete", "random_mask", "no_noise", "full_mask"])
FREEZE_MODULE = ChoiceEnum(["reorder", "translator","None"])
EVAL_BLEU_ORDER = 4
logger = logging.getLogger(__name__)

@dataclass
class TranslationAlignReorderConfig(TranslationConfig):
    noise: NOISE_CHOICES = field(
        default="random_delete",
        metadata={"help": "type of noise"},
    )

    random_mask_rate: float = field(
        default=0.2, metadata={"help": "The multiplier value of the source upsampling "},
    )   
    freeze_module: FREEZE_MODULE = field(
        default="None",
        metadata={"help": "choose a module to freeze"},
    )
    add_blank_symbol: bool = field(
        default=False, metadata={"help": "add the blank symbol in the target dictionary"},
    )     
    prepend_bos: bool = field(
        default=True, metadata={"help": "if set, without bos token"},
    )      
       
    global_token: bool = field(
        default=False, metadata={"help": "if set, use global_token but not calculate loss in nat_ctc_loss"},
    )     

    iterative_reorder_translator: bool = field(
        default=False, metadata={"help": "if set, iterative training reorder and translator"},
    )     
    
    iter_num_reorder: int = field(
        default=2, metadata={"help": "reorder traning after the numbers of translator  "},
    )  
    align_position_pad_index: int = field(
        default=513, metadata={"help": "position padding index, defult is 0 "},
    )      
    # args for pretrained models:
    pretrained_lm_name: str = field(
        default="None", metadata={"help": "Name of the path for the LM model"},
    )  
    lm_loss_dis: bool = field(
        default=False, metadata={"help": "compute LM loss using distribution"},
    )  
    lm_loss_layer: int = field(
        default=-1, metadata={"help": "the lm loss layer , default is -1 (-1 means last layer)"},
    )       
    no_atten_mask: bool = field(
        default=False, metadata={"help": "the model attention mask is None"},
    )
    debug: bool = field(
        default=False, metadata={"help": "debug"},
    )  
    twcc: bool = field(
        default=False, metadata={"help": "work on twcc while watch_test_bleu"},
    )
    watch_test_bleu: bool = field(
        default=False, metadata={"help": "watch test bleu while valid process"},
    )    
                              
    # ctc_beam_decoding
    ctc_beam_decoding: bool = field(
        default=False, metadata={"help": "use ctc beam decoding"},
    ) 
    beam_size: int = field(
        default=1, metadata={"help": "beam size for ctc beam decoding"},
    )
    kenlm_path: str = field(
        default="None", metadata={"help": "path to the kenlm model for ctc beam decoding"},
    ) 
    alpha: float = field(
        default=0.0, metadata={"help": "alpha for ctc beam decoding"},
    )
    beta: float = field(
        default=0.0, metadata={"help": "beta for ctc beam decoding"},
    )   
    pretrained_lm_path: str = field(
        default="None", metadata={"help": "path for the exist pretrained lm model, like pruned model "},
    )     
    pretrained_model_path: str = field(
        default="None", metadata={"help": "path for the exist pretrained model, like pruned model "},
    )            

@register_task("translation_align_reorder", dataclass=TranslationAlignReorderConfig)
class TranslationaAlignReorder(TranslationTask):
    """
    Translation (Sequence Generation) task for Levenshtein Transformer
    See `"Levenshtein Transformer" <https://arxiv.org/abs/1905.11006>`_.
    """
    cfg: TranslationAlignReorderConfig

    def __init__(self, cfg: TranslationAlignReorderConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        if cfg.add_blank_symbol :
            self.blank_symbol = '<blank>'
            self.tgt_dict.add_symbol(self.blank_symbol)
            self.src_dict.add_symbol(self.blank_symbol)
            def blank_symbol(self):
                return self.blank_symbol 
        self.prepend_bos = cfg.prepend_bos
        self.freeze_module = self.cfg.freeze_module
        self.iter_num_reorder = cfg.iter_num_reorder 
        self.iterative_reorder_translator = cfg.iterative_reorder_translator
        self.switch = True
        self.align_position_pad_index = cfg.align_position_pad_index

        if cfg.pretrained_lm_name == "None" :
            self.pretrained_lm = None
        else:
            if cfg.pretrained_lm_path != "None" :
                from transformers import AutoConfig
                import os        
                lm_config = AutoConfig.from_pretrained(os.path.join(cfg.pretrained_lm_path,"config.json"))
                # self.pretrained_lm = AutoModel.from_pretrained(os.path.join(cfg.pretrained_lm_path,"pytorch_model.bin"), config=lm_config)
                self.pretrained_lm = AutoModelForMaskedLM.from_pretrained(os.path.join(cfg.pretrained_lm_path,"pytorch_model.bin"), config=lm_config)
            else:
                if cfg.lm_loss_dis :
                    self.pretrained_lm = AutoModelForMaskedLM.from_pretrained(cfg.pretrained_lm_name)
                else:
                    self.pretrained_lm = AutoModel.from_pretrained(cfg.pretrained_lm_name)             
        
        
        
        
        self.lm_loss_layer = cfg.lm_loss_layer
        self.twcc = cfg.twcc
        self.watch_test_bleu = cfg.watch_test_bleu

    # class Identity(torch.nn.Module):
    #     def __init__(self):
    #         super(torch.nn.Identity, self).__init__()
            
    #     def forward(self, x):
    #         return x

  
    @classmethod
    def load_dictionary(cls, filename, vocab_file):
        """Load the dictionary from the filename
        Args:
            filename (str): the filename
        """
        return Dictionary.load(filename, vocab_file)

    @classmethod
    def build_dictionary(
        cls, filenames, args, tgt, workers=1, threshold=-1, nwords=-1, padding_factor=8
    ):
        """Build the dictionary
        Args:
            filenames (list): list of filenames
            workers (int): number of concurrent workers
            threshold (int): defines the minimum word count
            nwords (int): defines the total number of words in the final dictionary,
                including special symbols
            padding_factor (int): can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        if tgt and args.no_vocab_for_tgt:
            d = Dictionary(vocab_file=None)
        else:
            d = Dictionary(vocab_file=args.vocab_file)
        for filename in filenames:
            Dictionary.add_file_to_dictionary(
                filename, d, tokenizer.tokenize_line, workers
            )
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d


    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        # infer langcodes
        src, tgt = self.cfg.source_lang, self.cfg.target_lang
        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            prepend_bos=self.prepend_bos,
            load_alignments=True,
            align_position_pad_index=self.align_position_pad_index
        )

    def inject_noise(self, target_tokens):
        def _random_delete(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(bos) | target_tokens.eq(eos), 0.0
            )
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True
            )

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = (
                2
                + (
                    (target_length - 2)
                    * target_score.new_zeros(target_score.size(0), 1).uniform_()
                ).long()
            )
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = (
                target_tokens.gather(1, target_rank)
                .masked_fill_(target_cutoff, pad)
                .gather(1, target_rank.masked_fill_(target_cutoff, max_len).sort(1)[1])
            )
            prev_target_tokens = prev_target_tokens[
                :, : prev_target_tokens.ne(pad).sum(1).max()
            ]

            return prev_target_tokens

        def _random_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = (
                target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
            )
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()*self.cfg.random_mask_rate
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk
            )
            return prev_target_tokens

        def _full_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_mask = (
                target_tokens.eq(bos) | target_tokens.eq(eos) | target_tokens.eq(pad)
            )
            return target_tokens.masked_fill(~target_mask, unk)

        if self.cfg.noise == "random_delete":
            return _random_delete(target_tokens)
        elif self.cfg.noise == "random_mask":
            return _random_mask(target_tokens)
        elif self.cfg.noise == "full_mask":
            return _full_mask(target_tokens)
        elif self.cfg.noise == "no_noise":
            return target_tokens
        else:
            raise NotImplementedError

    def build_generator(self, models, args, **unused):
        # add models input to match the API for SequenceGenerator
        from fairseq.nat_encoder_generator import NATEncoderGenerator
        return NATEncoderGenerator(
            self.target_dictionary,
            self.source_dictionary,
            eos_penalty=getattr(args, "iter_decode_eos_penalty", 0.0),
            max_iter=getattr(args, "iter_decode_max_iter", 1),
            beam_size=getattr(args, "iter_decode_with_beam", 1),
            reranking=getattr(args, "iter_decode_with_external_reranker", False),
            decoding_format=getattr(args, "decoding_format", None),
            adaptive=not getattr(args, "iter_decode_force_max_iter", False),
            retain_history=getattr(args, "retain_iter_history", False),
        )
    
    def build_bpe(self, args, if_src=False):
        """Build the tokenizer for this task."""
        return encoders.build_bpe(args, if_src)
    
    def build_model(self, cfg, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)
        if self.cfg.eval_bleu:
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
            )

            gen_args = json.loads(self.cfg.eval_bleu_args)
            self.nat_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
    
        return model


    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            # Though see Susanto et al. (ACL 2020): https://www.aclweb.org/anthology/2020.acl-main.325/
            raise NotImplementedError(
                "Constrained decoding with the translation_lev task is not supported"
            )

        return LanguagePairDataset(
            src_tokens, src_lengths, self.source_dictionary, append_bos=True
        )
    
    def pad_alignment_position(values, pad_idx):
        size = values.size()

        batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
        res = values.new().fill_(pad_idx)

        def copy_tensor(src, dst):
            dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][: len(v)])        


    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):  
        model.train()
        # if self.iterative_reorder_translator :             
        #     if update_num % self.iter_num_reorder == 0 :                 
        #         self.switch = not self.switch    
        #         if self.switch :
        #             print("Freeze module : reorder")
        #         else:
        #             print("Freeze module : translator")
                              
        #     if self.switch :                 
        #         self.freeze_module =  'reorder'             
        #     else :                 
        #         self.freeze_module = 'translator'        
            
        """ for merge only use
        cuda0 = torch.device('cuda:0')
        loss=torch.Tensor(0)
        sample_size=1
        logging_output={'loss': 0, 'sample_size' : 1}#, 'nll_loss': torch.Tensor(0.187), 'ntokens': 960, 'nsentences': 40, 'sample_size': 1, 'word_ins-loss': 0.18786469101905823}
        return loss, sample_size, logging_output
        """
        loss, sample_size, logging_output = criterion(model, sample, update_num,
                                                      self.pretrained_lm, self.lm_loss_layer)
         
        if ignore_grad:
            loss *= 0

        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion, update_num):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample, update_num)
        # return loss, sample_size, logging_output

        if self.cfg.eval_bleu:
            bleu = self._inference_with_bleu(self.nat_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output        

    def upsampling(self, source, rate): 
        upsampled =  torch.repeat_interleave(source, rate, dim=1)
        return upsampled

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.cfg.eval_bleu:

            def sum_logs(key):
                import torch

                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect

                    try:
                        from sacrebleu.metrics import BLEU

                        comp_bleu = BLEU.compute_bleu
                    except ImportError:
                        # compatibility API for sacrebleu 1.x
                        import sacrebleu

                        comp_bleu = sacrebleu.compute_bleu

                    fn_sig = inspect.getfullargspec(comp_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = comp_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=int(meters["_bleu_sys_len"].sum),
                        ref_len=int(meters["_bleu_ref_len"].sum),
                        **smooth,
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)   
                
                     
    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s
        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            if len(gen_out[i][0]["tokens"]) > 0 :
                remove_duplicate_tokens = torch.unique_consecutive(gen_out[i][0]["tokens"])
            else:
                remove_duplicate_tokens = gen_out[i][0]["tokens"]

            hyps.append(decode(remove_duplicate_tokens))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.cfg.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])
     
    