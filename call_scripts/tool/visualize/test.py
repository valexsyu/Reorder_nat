
import os
from transformers import AutoModel, AutoModelForMaskedLM, AutoConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

path='/livingrooms/valexsyu/dataset/model/mbert/pruned_models_BertForMaskedLM/pruned_V26458'
translator_config = AutoConfig.from_pretrained(os.path.join(path,"config.json"))
translator = AutoModelForMaskedLM.from_pretrained(os.path.join(path,"pytorch_model.bin"), config=translator_config)
import pdb;pdb.set_trace()
print(translator)