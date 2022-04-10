# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

logits = torch.randn(3, 10)
print(logits)
sampled_hard=F.gumbel_softmax(logits, tau=1, hard=True)
sampled_soft = F.gumbel_softmax(logits, tau=1, hard=False)
print(sampled_hard)
print(sampled_soft)






