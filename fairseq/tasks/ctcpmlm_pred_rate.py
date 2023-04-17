# valex 2323/4/23


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
from fairseq.tasks.translation_align_reorder import (
    TranslationAlignReorderConfig,
    TranslationaAlignReorder,
)


logger = logging.getLogger(__name__)


@dataclass
class CTCPMLMPredRateConfig(TranslationAlignReorderConfig):
    use_initial_target_rate: bool = field(
        default=False, metadata={"help": "use the inital pred rate to be target"},
    )   
    initial_target_rate_value: float = field(
        default=2.0, metadata={"help": "use the inital pred rate to be target"},
    )     


@register_task("ctcpmlm_pred_rate", dataclass=TranslationAlignReorderConfig)
class CTCPMLMPredRate(TranslationaAlignReorder):
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):  
        if update_num <= 1
    