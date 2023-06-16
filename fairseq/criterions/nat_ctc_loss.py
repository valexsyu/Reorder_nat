# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from torch import Tensor
from fairseq.criterions.nat_loss import (
    LabelSmoothedDualImitationCriterion, 
    LabelSmoothedDualImitationCriterionConfig,
)
from fairseq.criterions import nat_loss 
from dataclasses import dataclass, field





@register_criterion("nat_ctc_loss", dataclass=LabelSmoothedDualImitationCriterionConfig)
class NatEncoderCTCLoss(LabelSmoothedDualImitationCriterion):

    def __init__(self, task, label_smoothing):
        super().__init__(task, label_smoothing)
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()    
        self.bos_idx = task.target_dictionary.bos()
        self.debug = task.cfg.debug  

        if task.cfg.blank_use_mask :
            if '[MASK]' in task.target_dictionary.symbols :
                self.blank_idx = task.target_dictionary.indices['[MASK]']
            elif '<mask>' in task.target_dictionary.symbols :
                self.blank_idx = task.target_dictionary.indices['<mask>']            
            else :
                import pdb;pdb.set_trace()
                print("check the MASK token symbol")
        else:
            self.blank_idx = (
                task.target_dictionary.index(task.blank_symbol)
                if hasattr(task, "blank_symbol")
                else self.bos_idx
            )  
            
        
        
    @classmethod
    def add_args(cls, parser):
        LabelSmoothedDualImitationCriterion.add_args(parser)
       
    
    def _compute_ctc_loss(  #valex
        self, lprobs, targets, masks=None, num_upsampling_rate=2, name="loss", factor=1.0, sample=None, reduction="mean"
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """
        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )
        
        
        lprobs = lprobs.contiguous()
        # lprobs = model.get_normalized_probs(
        #     [outputs], log_probs=True
        # ).contiguous()  # (T, B, C) from the encoder 
        
        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
            input_lengths_upsample = (num_upsampling_rate*input_lengths).type_as(input_lengths) 
        else:
            input_lengths = lprobs.new_full(
                (lprobs.size(1),), lprobs.size(0), dtype=torch.long
            )
        
        pad_mask = (targets != self.pad_idx) & (
                    targets != self.eos_idx) & (
                    targets != self.bos_idx)
                    
        targets_flat = targets.masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        lprobs = lprobs.transpose(0,1)    

 
        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths_upsample,
                target_lengths,
                blank=self.blank_idx, 
                reduction=reduction,
                zero_infinity=True,
            )           
            
        loss = loss * factor
        nll_loss = loss
        
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}
    
    def _compute_loss(
        self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")

            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device), reduction="none")
                losses = losses.sum(-1)

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = (
                    nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                )
            else:
                loss = nll_loss

        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

    def _compute_mse_loss(
        self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        
        nll_loss = F.mse_loss(outputs,targets,reduction='sum')
        loss = nll_loss

        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

    def _compute_ce_loss(
        self, outputs, targets, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
        outputs: batch x len 
        targets: batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )


        if targets.dim() == 1:
            losses = F.nll_loss(outputs, targets.to(outputs.device), reduction="none")

        else:  # soft-labels
            losses = F.kl_div(outputs, targets.to(outputs.device), reduction="none")
            losses = losses.sum(-1)

        nll_loss = mean_ds(losses)
        if label_smoothing > 0:
            loss = (
                nll_loss * (1 - label_smoothing) - mean_ds(outputs) * label_smoothing
            )
        else:
            loss = nll_loss

        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}  

    def forward(self, model, sample, update_num , pretrained_lm=None, lm_loss_layer=-1, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        #tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]
        if sample.get("alignments", None) is not None: 
            tgt_tokens , alignments= sample["target"], sample["alignments"]
        else:
            tgt_tokens = sample["target"]
            alignments = None
        outputs = model(src_tokens, src_lengths, tgt_tokens, alignments, update_num, pretrained_lm, lm_loss_layer)
        
        losses, nll_loss = [], []

        for obj in outputs:
            if outputs[obj].get("loss_type", "CTC") == "CTC":
                _losses = self._compute_ctc_loss(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask", None),
                    outputs[obj].get("num_upsampling_rate", 2), 
                    name=obj + "-loss",
                    factor=1.0,
                    sample=sample,
                    
                )
            elif outputs[obj].get("loss_type", "CTC") == "MSE":
                _losses = self._compute_mse_loss(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask", None),
                    outputs[obj].get("ls", 0.0),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )     
            else:
                _losses = self._custom_loss(
                    outputs[obj].get("loss"),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )

            losses += [_losses]
            if outputs[obj].get("nll_loss", False):
                nll_loss += [_losses.get("nll_loss", 0.0)]
        loss = sum(l["loss"] for l in losses)
        nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 else loss.new_tensor(0)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1        
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))

        metrics.log_scalar(
            "loss", loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
    
    
    
@register_criterion("nat_ctc_sel_rate_loss", dataclass=LabelSmoothedDualImitationCriterionConfig)
class NatCTCSelRateLoss(NatEncoderCTCLoss):
    def __init__(self, task, label_smoothing):
        super().__init__(task, label_smoothing)
        self.rate_list = task.cfg.rate_list
        
        self.max_update = task.cfg.max_update
        self.lmax_only_step = task.cfg.lmax_only_step
    def _compute_ctc_loss(  #valex
        self, lprobs, targets, masks=None, num_upsampling_rate=2, name="loss", factor=1.0, sample=None, reduction="mean"
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """
        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )
        
        
        lprobs = lprobs.contiguous()
        # lprobs = model.get_normalized_probs(
        #     [outputs], log_probs=True
        # ).contiguous()  # (T, B, C) from the encoder 
        
        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
            input_lengths_upsample = (num_upsampling_rate*input_lengths).type_as(input_lengths) 
        else:
            input_lengths = lprobs.new_full(
                (lprobs.size(1),), lprobs.size(0), dtype=torch.long
            )
        
        pad_mask = (targets != self.pad_idx) & (
                    targets != self.eos_idx) & (
                    targets != self.bos_idx)
                    
        targets_flat = targets.masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        lprobs = lprobs.transpose(0,1)    

        
        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths_upsample,
                target_lengths,
                blank=self.blank_idx, 
                reduction=reduction,
                zero_infinity=True,
            )        

        loss = loss * factor
        nll_loss = loss
        
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor, "tgt_lengths":target_lengths}
    
    def loss_collection(self, model_out, pve_losses, collection_name, sample, ctc_losses, ce_losses, others_losses): 
        losses = []
        for obj in model_out:
            if model_out[obj].get("loss_type", "CTC") == "CTC":
                _losses = self._compute_ctc_loss(
                    model_out[obj].get("out"),
                    model_out[obj].get("tgt"),
                    model_out[obj].get("mask", None),
                    model_out[obj].get("num_upsampling_rate", 2), 
                    name=obj + "-loss",
                    factor=1.0,
                    sample=sample,
                    reduction='none',
                )   
                ctc_losses += [_losses['loss']]  
            elif model_out[obj].get("loss_type", "CTC") == "CE":
                _losses = self._compute_ce_loss(
                    model_out[obj].get("out"),
                    model_out[obj].get("tgt"),
                    model_out[obj].get("ls", 0.0),
                    name=obj + "-loss",
                    factor=model_out[obj].get("factor", 1.0),
                )       
                ce_losses += [_losses['loss']]                      
            else:
                _losses = self._custom_loss(
                    model_out[obj].get("loss"),
                    name=obj + "-loss",
                    factor=model_out[obj].get("factor", 1.0),
                )        
                others_losses += [_losses['loss']] 
            
            
            losses += [_losses]  
        
        pve_losses[collection_name] = losses
         
        return pve_losses, ctc_losses, ce_losses, others_losses
    
    
    
    def forward(self, model, sample, update_num , pretrained_lm=None, lm_loss_layer=-1, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
    
        
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        #tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]
        if sample.get("alignments", None) is not None: 
            tgt_tokens , alignments= sample["target"], sample["alignments"]
        else:
            tgt_tokens = sample["target"]
            alignments = None
        

        
        
        # for upsampling_rate in self.rate_list : 
        
        collect_losses={}
        ctc_losses = []
        ce_losses = []
        others_losses = []
        for upsampling_rate in self.rate_list :             
            outputs = model(src_tokens, src_lengths, tgt_tokens, alignments, update_num, 
                            pretrained_lm, lm_loss_layer, upsampling_rate)
            
            ## ex : collect_losses = {2:[{ctc-loss},{ce-loss}], 3:[{},{}], 4[{},{}] }
            collection_name = "r-" + str(upsampling_rate)
            collect_losses, ctc_losses, _ , others_losses = self.loss_collection(outputs, collect_losses, collection_name, sample, 
                                                                   ctc_losses, ce_losses, others_losses) 
        if len(ctc_losses) > 0 :
            ctc_losses = torch.stack(ctc_losses)
        if len(others_losses) > 0 :
            others_losses = torch.stack(others_losses)
         
        tgt_lengths=collect_losses[list(collect_losses.keys())[0]][0]['tgt_lengths'].detach() 
        sum_tgt_lengths = torch.sum(tgt_lengths)
        num_rate_list = len(self.rate_list)
        avg_tgt_lengths = sum_tgt_lengths/num_rate_list
        #leave some steps for checkpoint averaging
        time = update_num / (self.max_update - self.lmax_only_step)
        curr_lambda = 1/3 ##2 use 2/3 and 3 use 3/3
        num_rate, bz = ctc_losses.size() # num_rate x bz size
        if time < curr_lambda:   
            t_1 = time / curr_lambda
            ctc_sum_loss = torch.sum(ctc_losses).div(sum_tgt_lengths).div(num_rate_list)
            ctc_lse_loss = - torch.sum(torch.logsumexp(-ctc_losses, dim = 0)).div(sum_tgt_lengths).div(num_rate_list)
            # ctc_sum_loss = ctc_losses.mean()  # bz size
            # ctc_lse_loss = - torch.sum(torch.logsumexp(-ctc_losses, dim = 0)) / bz
            loss = t_1 * ctc_lse_loss + (1 - t_1) * ctc_sum_loss                    
        elif time < 1:
            t_2 = (time - curr_lambda) / (1 - curr_lambda)
            ctc_lse_loss = - torch.sum(torch.logsumexp(-ctc_losses, dim = 0)).div(sum_tgt_lengths).div(num_rate_list)
            ctc_min_loss, min_idx = torch.min(ctc_losses, dim = 0)
            ctc_min_loss = torch.sum(ctc_min_loss).div(sum_tgt_lengths)  
            # ctc_lse_loss = - torch.sum(torch.logsumexp(-ctc_losses, dim = 0)) / bz
            # ctc_min_loss, min_idx = torch.min(ctc_losses, dim = 0)
            # ctc_min_loss = ctc_min_loss.mean()    
            loss = t_2 * ctc_min_loss + (1 - t_2) * ctc_lse_loss 
        else:
            ctc_min_loss, min_idx = torch.min(ctc_losses, dim = 0)
            ctc_min_loss = torch.sum(ctc_min_loss).div(sum_tgt_lengths) 
            # ctc_min_loss, min_idx = torch.min(ctc_losses, dim = 0)
            # ctc_min_loss = ctc_min_loss.mean()    
            loss = ctc_min_loss                       
              

        nll_loss = loss.new_tensor(0)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1        
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        for collection_name, losses in collect_losses.items() :
            for l in losses:
                logging_output[collection_name + "-"+ l["name"]] = (
                    utils.item(l["loss"].mean().data / l["factor"])
                    if reduce
                    else l[["loss"]].data / l["factor"]
                )

        return loss, sample_size, logging_output





@register_criterion("nat_ctc_pred_rate_loss", dataclass=LabelSmoothedDualImitationCriterionConfig)
class NatCTCPredRateLoss(NatCTCSelRateLoss):
    def __init__(self, task, label_smoothing):
        super().__init__(task, label_smoothing)
        

    def forward(self, model, sample, update_num , pretrained_lm=None, lm_loss_layer=-1, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
    
        
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        #tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]
        if sample.get("alignments", None) is not None: 
            tgt_tokens , alignments= sample["target"], sample["alignments"]
        else:
            tgt_tokens = sample["target"]
            alignments = None
        

        
        
        # for upsampling_rate in self.rate_list : 
        
        collect_losses={}
        ctc_losses = []
        ce_losses = []
        others_losses = []
        for upsampling_rate in self.rate_list :      
            outputs = model.ctc_forward(src_tokens, src_lengths, tgt_tokens, alignments, update_num, 
                            pretrained_lm, lm_loss_layer, upsampling_rate)
            
            ## ex : collect_losses = {2:[{ctc-loss},{ce-loss}], 3:[{},{}], 4[{},{}] }
            collection_name = "r-" + str(upsampling_rate)
            collect_losses, ctc_losses, _ , others_losses = self.loss_collection(outputs, collect_losses, collection_name , sample, 
                                                                   ctc_losses, ce_losses, others_losses) 
       
        if len(ctc_losses) > 0 :
            ctc_losses = torch.stack(ctc_losses)
        if len(others_losses) > 0 :
            others_losses = torch.stack(others_losses)
        
        tgt_rate = F.softmax(-ctc_losses.transpose(0,1), dim=1).detach()
        
        rate_outputs = model.rate_pred_forward(src_tokens, tgt_rate)
        collection_name = "rate"
        collect_losses, _ , ce_losses , _  = self.loss_collection(rate_outputs, collect_losses, "rate", sample, 
                                                                   ctc_losses, ce_losses, others_losses) 
        
        ce_loss = ce_losses[0]
        #leave some steps for checkpoint averaging
        
        tgt_lengths=collect_losses[list(collect_losses.keys())[0]][0]['tgt_lengths']
        sum_tgt_lengths = torch.sum(tgt_lengths)
        num_rate_list = len(self.rate_list)
        time = update_num / (self.max_update - self.lmax_only_step)
        curr_lambda = 1/3
        num_rate, bz = ctc_losses.size() # num_rate x bz size
        if time < curr_lambda:   
            t_1 = time / curr_lambda
            ctc_avg_loss = torch.sum(ctc_losses).div(sum_tgt_lengths).div(num_rate_list)
            # ctc_avg_loss = ctc_losses.mean()  # bz size
            ctc_loss = ctc_avg_loss
            ce_loss = ce_loss
        elif time < 1:
            t_2 = (time - curr_lambda) / (1 - curr_lambda)
            rate_max_lprob, max_idx = torch.max(rate_outputs['rate']['out'].detach(), dim = 1)
            ctc_rmax_losses = torch.gather(ctc_losses, 0, max_idx.view(1, -1))
            ctc_rmax_loss = torch.sum(ctc_rmax_losses).div(sum_tgt_lengths)
            ctc_avg_loss = torch.sum(ctc_losses).div(sum_tgt_lengths).div(num_rate_list)
            # ctc_rmax_loss = ctc_rmax_losses.mean()    
            # ctc_avg_loss = ctc_losses.mean()
            ctc_loss = t_2 * ctc_rmax_loss + (1 - t_2) * ctc_avg_loss    
            ce_loss = ce_loss
        else:
            rate_max_lprob, max_idx = torch.max(rate_outputs['rate']['out'].detach(), dim = 1)
            ctc_rmax_losses = torch.gather(ctc_losses, 0, max_idx.view(1, -1))
            ctc_rmax_loss = torch.sum(ctc_rmax_losses).div(sum_tgt_lengths)
            # ctc_rmax_loss = ctc_rmax_losses.mean()  
            ctc_loss = ctc_rmax_loss    
            ce_loss = ce_loss
              

        

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1        
        logging_output = {
            "ctc_loss": ctc_loss.data,
            "rate_loss": ce_loss.data,
            "nll_loss": 0,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        for collection_name, losses in collect_losses.items() :
            for l in losses:
                logging_output[collection_name + "-" + l["name"]] = (
                    utils.item(l["loss"].mean().data / l["factor"])
                    if reduce
                    else l[["loss"]].data / l["factor"]
                )

        return ctc_loss, ce_loss, sample_size, logging_output





@register_criterion("nat_ctc_avg_rate_loss", dataclass=LabelSmoothedDualImitationCriterionConfig)
class NatCTCAvgRateLoss(NatEncoderCTCLoss):
    def __init__(self, task, label_smoothing):
        super().__init__(task, label_smoothing)
        self.rate_list = task.cfg.rate_list
        self.rate_weight_list = task.cfg.rate_weight_list
        self.max_update = task.cfg.max_update
        self.lmax_only_step = task.cfg.lmax_only_step
    def _compute_ctc_loss(  #valex
        self, lprobs, targets, masks=None, num_upsampling_rate=2, name="loss", factor=1.0, sample=None, reduction="mean"
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """
        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )
        
        
        lprobs = lprobs.contiguous()
        # lprobs = model.get_normalized_probs(
        #     [outputs], log_probs=True
        # ).contiguous()  # (T, B, C) from the encoder 
        
        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
            input_lengths_upsample = (num_upsampling_rate*input_lengths).type_as(input_lengths) 
        else:
            input_lengths = lprobs.new_full(
                (lprobs.size(1),), lprobs.size(0), dtype=torch.long
            )
        
        pad_mask = (targets != self.pad_idx) & (
                    targets != self.eos_idx) & (
                    targets != self.bos_idx)
                    
        targets_flat = targets.masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        lprobs = lprobs.transpose(0,1)    

        
        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths_upsample,
                target_lengths,
                blank=self.blank_idx, 
                reduction=reduction,
                zero_infinity=True,
            )        

        loss = loss * factor
        nll_loss = loss
        
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor, "tgt_lengths":target_lengths}
    
    def loss_collection(self, model_out, pve_losses, collection_name, sample, ctc_losses, ce_losses, others_losses, factor=1): 
        losses = []
        for obj in model_out:
            if model_out[obj].get("loss_type", "CTC") == "CTC":
                _losses = self._compute_ctc_loss(
                    model_out[obj].get("out"),
                    model_out[obj].get("tgt"),
                    model_out[obj].get("mask", None),
                    model_out[obj].get("num_upsampling_rate", 2), 
                    name=obj + "-loss",
                    factor=factor,
                    sample=sample,
                    reduction='none',
                )   
                ctc_losses += [_losses['loss']]  
            elif model_out[obj].get("loss_type", "CTC") == "CE":
                _losses = self._compute_ce_loss(
                    model_out[obj].get("out"),
                    model_out[obj].get("tgt"),
                    model_out[obj].get("ls", 0.0),
                    name=obj + "-loss",
                    factor=model_out[obj].get("factor", 1.0),
                )       
                ce_losses += [_losses['loss']]                      
            else:
                _losses = self._custom_loss(
                    model_out[obj].get("loss"),
                    name=obj + "-loss",
                    factor=model_out[obj].get("factor", 1.0),
                )        
                others_losses += [_losses['loss']] 
            
            
            losses += [_losses]  
        
        pve_losses[collection_name] = losses
         
        return pve_losses, ctc_losses, ce_losses, others_losses
    
    
    
    def forward(self, model, sample, update_num , pretrained_lm=None, lm_loss_layer=-1, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
    
        
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        #tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]
        if sample.get("alignments", None) is not None: 
            tgt_tokens , alignments= sample["target"], sample["alignments"]
        else:
            tgt_tokens = sample["target"]
            alignments = None
        

        
        
        # for upsampling_rate in self.rate_list : 
        
        collect_losses={}
        ctc_losses = []
        ce_losses = []
        others_losses = []
        for i , rate_and_weight in enumerate(zip(self.rate_list, self.rate_weight_list)) :   
            upsampling_rate = rate_and_weight[0]
            factor = rate_and_weight[1]       
            outputs = model(src_tokens, src_lengths, tgt_tokens, alignments, update_num, 
                            pretrained_lm, lm_loss_layer, upsampling_rate)
            
            ## ex : collect_losses = {2:[{ctc-loss},{ce-loss}], 3:[{},{}], 4[{},{}] }
            collection_name = "r-" + str(upsampling_rate)
            collect_losses, ctc_losses, _ , others_losses = self.loss_collection(outputs, collect_losses, collection_name, sample, 
                                                                   ctc_losses, ce_losses, others_losses, factor) 
        if len(ctc_losses) > 0 :
            ctc_losses = torch.stack(ctc_losses)
        if len(others_losses) > 0 :
            others_losses = torch.stack(others_losses)
         
        tgt_lengths=collect_losses[list(collect_losses.keys())[0]][0]['tgt_lengths'].detach() 
        sum_tgt_lengths = torch.sum(tgt_lengths)
        num_rate_list = len(self.rate_list)
        avg_tgt_lengths = sum_tgt_lengths/num_rate_list
        #leave some steps for checkpoint averaging
        time = update_num / (self.max_update - self.lmax_only_step)
        curr_lambda = 1/3 ##2 use 2/3 and 3 use 3/3
        num_rate, bz = ctc_losses.size() # num_rate x bz size
        loss = torch.sum(ctc_losses).div(sum_tgt_lengths).div(num_rate_list)                    
            
        nll_loss = loss.new_tensor(0)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1        
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        for collection_name, losses in collect_losses.items() :
            for l in losses:
                logging_output[collection_name + "-"+ l["name"]] = (
                    utils.item(l["loss"].mean().data / l["factor"])
                    if reduce
                    else l[["loss"]].data / l["factor"]
                )

        return loss, sample_size, logging_output
    
    
    
@register_criterion("nat_ctc_predsel_rate_loss", dataclass=LabelSmoothedDualImitationCriterionConfig)
class NatCTCPredSelRateLoss(NatCTCSelRateLoss):
    def __init__(self, task, label_smoothing):
        super().__init__(task, label_smoothing)
        

    def forward(self, model, sample, update_num , pretrained_lm=None, lm_loss_layer=-1, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
    
        
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        #tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]
        if sample.get("alignments", None) is not None: 
            tgt_tokens , alignments= sample["target"], sample["alignments"]
        else:
            tgt_tokens = sample["target"]
            alignments = None
        

        
        
        # for upsampling_rate in self.rate_list : 
        
        collect_losses={}
        ctc_losses = []
        ce_losses = []
        others_losses = []
        for upsampling_rate in self.rate_list :      
            outputs = model.ctc_forward(src_tokens, src_lengths, tgt_tokens, alignments, update_num, 
                            pretrained_lm, lm_loss_layer, upsampling_rate)
            
            ## ex : collect_losses = {2:[{ctc-loss},{ce-loss}], 3:[{},{}], 4[{},{}] }
            collection_name = "r-" + str(upsampling_rate)
            collect_losses, ctc_losses, _ , others_losses = self.loss_collection(outputs, collect_losses, collection_name , sample, 
                                                                   ctc_losses, ce_losses, others_losses) 
       
        if len(ctc_losses) > 0 :
            ctc_losses = torch.stack(ctc_losses)
        if len(others_losses) > 0 :
            others_losses = torch.stack(others_losses)
        
        tgt_rate = F.softmax(-ctc_losses.transpose(0,1), dim=1).detach()
        
        rate_outputs = model.rate_pred_forward(src_tokens, tgt_rate)
        collection_name = "rate"
        collect_losses, _ , ce_losses , _  = self.loss_collection(rate_outputs, collect_losses, "rate", sample, 
                                                                   ctc_losses, ce_losses, others_losses) 
        
        ce_loss = ce_losses[0]
        #leave some steps for checkpoint averaging
        
        tgt_lengths=collect_losses[list(collect_losses.keys())[0]][0]['tgt_lengths']
        sum_tgt_lengths = torch.sum(tgt_lengths)
        num_rate_list = len(self.rate_list)
        time = update_num / (self.max_update - self.lmax_only_step)
        curr_lambda = 1/3
        num_rate, bz = ctc_losses.size() # num_rate x bz size
        if time < curr_lambda:   
            t_1 = time / curr_lambda
            ctc_avg_loss = torch.sum(ctc_losses).div(sum_tgt_lengths).div(num_rate_list)
            ctc_lse_loss = - torch.sum(torch.logsumexp(-ctc_losses, dim = 0)).div(sum_tgt_lengths).div(num_rate_list)
            # ctc_avg_loss = ctc_losses.mean()  # bz size
            ctc_loss = t_1 * ctc_lse_loss + (1 - t_1) * ctc_avg_loss  
            ce_loss = ce_loss
        elif time < 1:
            t_2 = (time - curr_lambda) / (1 - curr_lambda)
            rate_max_lprob, max_idx = torch.max(rate_outputs['rate']['out'].detach(), dim = 1)
            ctc_rmax_losses = torch.gather(ctc_losses, 0, max_idx.view(1, -1))
            ctc_rmax_loss = torch.sum(ctc_rmax_losses).div(sum_tgt_lengths)
            ctc_lse_loss = - torch.sum(torch.logsumexp(-ctc_losses, dim = 0)).div(sum_tgt_lengths).div(num_rate_list)
            # ctc_rmax_loss = ctc_rmax_losses.mean()    
            # ctc_avg_loss = ctc_losses.mean()
            ctc_loss = t_2 * ctc_rmax_loss + (1 - t_2) * ctc_lse_loss    
            ce_loss = ce_loss
        else:
            rate_max_lprob, max_idx = torch.max(rate_outputs['rate']['out'].detach(), dim = 1)
            ctc_rmax_losses = torch.gather(ctc_losses, 0, max_idx.view(1, -1))
            ctc_rmax_loss = torch.sum(ctc_rmax_losses).div(sum_tgt_lengths)
            # ctc_rmax_loss = ctc_rmax_losses.mean()  
            ctc_loss = ctc_rmax_loss    
            ce_loss = ce_loss

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1        
        logging_output = {
            "ctc_loss": ctc_loss.data,
            "rate_loss": ce_loss.data,
            "nll_loss": 0,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        for collection_name, losses in collect_losses.items() :
            for l in losses:
                logging_output[collection_name + "-" + l["name"]] = (
                    utils.item(l["loss"].mean().data / l["factor"])
                    if reduce
                    else l[["loss"]].data / l["factor"]
                )

        return ctc_loss, ce_loss, sample_size, logging_output
    
    
    
    
    

@register_criterion("nat_ctc_predsel_rate_test_loss", dataclass=LabelSmoothedDualImitationCriterionConfig)
class NatCTCPredSelRateTestLoss(NatCTCSelRateLoss):
    def __init__(self, task, label_smoothing):
        super().__init__(task, label_smoothing)
        

    def forward(self, model, sample, update_num , pretrained_lm=None, lm_loss_layer=-1, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
    
        
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        #tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]
        if sample.get("alignments", None) is not None: 
            tgt_tokens , alignments= sample["target"], sample["alignments"]
        else:
            tgt_tokens = sample["target"]
            alignments = None
        

        
        
        # for upsampling_rate in self.rate_list : 
        
        collect_losses={}
        ctc_losses = []
        ce_losses = []
        others_losses = []
        for upsampling_rate in self.rate_list :      
            outputs = model.ctc_forward(src_tokens, src_lengths, tgt_tokens, alignments, update_num, 
                            pretrained_lm, lm_loss_layer, upsampling_rate)
            
            ## ex : collect_losses = {2:[{ctc-loss},{ce-loss}], 3:[{},{}], 4[{},{}] }
            collection_name = "r-" + str(upsampling_rate)
            collect_losses, ctc_losses, _ , others_losses = self.loss_collection(outputs, collect_losses, collection_name , sample, 
                                                                   ctc_losses, ce_losses, others_losses) 
       
        if len(ctc_losses) > 0 :
            ctc_losses = torch.stack(ctc_losses)
        if len(others_losses) > 0 :
            others_losses = torch.stack(others_losses)
        
        tgt_rate = F.softmax(-ctc_losses.transpose(0,1), dim=1).detach()
        
        rate_outputs = model.rate_pred_forward(src_tokens, tgt_rate)
        collection_name = "rate"
        collect_losses, _ , ce_losses , _  = self.loss_collection(rate_outputs, collect_losses, "rate", sample, 
                                                                   ctc_losses, ce_losses, others_losses) 
        
        ce_loss = ce_losses[0]
        #leave some steps for checkpoint averaging
        
        tgt_lengths=collect_losses[list(collect_losses.keys())[0]][0]['tgt_lengths']
        sum_tgt_lengths = torch.sum(tgt_lengths)
        num_rate_list = len(self.rate_list)
        time = update_num / (self.max_update - self.lmax_only_step)
        curr_lambda = 1/3
        num_rate, bz = ctc_losses.size() # num_rate x bz size
        if time < curr_lambda:   
            t_1 = time / curr_lambda
            ctc_avg_loss = torch.sum(ctc_losses).div(sum_tgt_lengths).div(num_rate_list)
            ctc_lse_loss = - torch.sum(torch.logsumexp(-ctc_losses, dim = 0)).div(sum_tgt_lengths).div(num_rate_list)
            # ctc_avg_loss = ctc_losses.mean()  # bz size
            ctc_loss = t_1 * ctc_lse_loss + (1 - t_1) * ctc_avg_loss  
            ce_loss = ce_loss
        elif time < 1:
            t_2 = (time - curr_lambda) / (1 - curr_lambda)
            rate_max_lprob, max_idx = torch.max(rate_outputs['rate']['out'].detach(), dim = 1)
            ctc_rmax_losses = torch.gather(ctc_losses, 0, max_idx.view(1, -1))
            ctc_rmax_loss = torch.sum(ctc_rmax_losses).div(sum_tgt_lengths)
            ctc_lse_loss = - torch.sum(torch.logsumexp(-ctc_losses, dim = 0)).div(sum_tgt_lengths).div(num_rate_list)
            # ctc_rmax_loss = ctc_rmax_losses.mean()    
            # ctc_avg_loss = ctc_losses.mean()
            ctc_loss = t_2 * ctc_rmax_loss + (1 - t_2) * ctc_lse_loss    
            ce_loss = ce_loss
        else:
            rate_max_lprob, max_idx = torch.max(rate_outputs['rate']['out'].detach(), dim = 1)
            ctc_rmax_losses = torch.gather(ctc_losses, 0, max_idx.view(1, -1))
            ctc_rmax_loss = torch.sum(ctc_rmax_losses).div(sum_tgt_lengths)
            # ctc_rmax_loss = ctc_rmax_losses.mean()  
            ctc_loss = ctc_rmax_loss    
            ce_loss = ce_loss

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1        
        logging_output = {
            "ctc_loss": ctc_loss.data,
            "rate_loss": ce_loss.data,
            "nll_loss": 0,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        for collection_name, losses in collect_losses.items() :
            for l in losses:
                logging_output[collection_name + "-" + l["name"]] = (
                    utils.item(l["loss"].mean().data / l["factor"])
                    if reduce
                    else l[["loss"]].data / l["factor"]
                )

        return ctc_loss, ce_loss, sample_size, logging_output    