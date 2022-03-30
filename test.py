# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import NATransformerDecoder, NATransformerModel , ensemble_decoder
from fairseq.models.transformer import Embedding
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from fairseq.models.roberta import RobertaModel


from transformers import BertTokenizer, AutoModel
from transformers import RobertaTokenizer, RobertaConfig
tokenizer = BertTokenizer.from_pretrained("jhu-clsp/bibert-ende")
model = AutoModel.from_pretrained("jhu-clsp/bibert-ende")



config = RobertaConfig.from_pretrained("jhu-clsp/bibert-ende")

roberta = torch.hub.load('pytorch/fairseq', 'roberta.base') 
print(roberta.named_modules)
print(model.named_modules)
import pdb;pdb.set_trace()
roberta.eval()  # disable dropout (or leave in train mode to finetune)





