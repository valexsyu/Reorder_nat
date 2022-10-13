
from transformers import AutoModel, AutoModelForMaskedLM
import torch
from fairseq import checkpoint_utils
from fairseq.models import (
    BaseFairseqModel,
)
PATH="/home/valexsyu/Doc/NMT/Reorder_nat/checkpoints/No-test/checkpoint_last.pt"

# model = AutoModelForMaskedLM.from_pretrained("jhu-clsp/bibert-ende")

class NATPretrainedModel(BaseFairseqModel):
    def __init__(self, translator):
        super().__init__()
        self.pretrained_lm = AutoModel.from_pretrained("jhu-clsp/bibert-ende") 
        self.translator = translator
        for params in self.pretrained_lm.parameters() :
            params.requires_grad=False                
        self.pretrained_lm.eval()         

    @classmethod
    def build_model(cls):
        """Build a new model instance."""
 
        translator = AutoModelForMaskedLM.from_pretrained("jhu-clsp/bibert-ende")

        return cls(translator)

model = NATPretrainedModel.build_model()

checkpoint_utils.torch_persistent_save(
    model.state_dict,
    PATH,
)
print(model)
# torch.save(model.state_dict(), PATH)