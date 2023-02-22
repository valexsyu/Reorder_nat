import torch
import torch.nn.functional as F
from fairseq.utils import new_arange, softmax, log_softmax
import evaluate
meteor = evaluate.load('meteor')