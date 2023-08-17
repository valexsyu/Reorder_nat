import torch
import torch.nn.functional as F
import pdb
import math
import numpy as np


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()

def dynamic_upsample_token(x, rate=2, insert_mask=False, insertion_position='uniform'):
    
    x = x.unsqueeze(0)
    
    B, L = x.size(0), x.size(1)
    new_length = torch.Tensor([L * rate]).int().item()
    # new_length = int(L * rate)   # 50*2.3=114.9999999
    pad = bos = eos = 0
    ### the mask is True when padding/bos/eos 

    mask = ~(
        x.ne(pad) & x.ne(bos) & x.ne(eos) # old version for eos is padding and masked
    )                     
                
    l = (x.new_ones(B, L) * rate).float()
    l = l.masked_fill(mask, 0)
    e = torch.cumsum(l, 1)
    c = e - l / 2
    t = e[:, -1].ceil().long()

    t = new_arange(t, new_length)[None, :].expand(l.size(0), -1)  # B x L2

    t_mask = t >= e[:, -1:]   # target padding mask
    
    if insertion_position == 'uniform':            
        w = -(t[:, None, :] - c[:, :, None]) ** 2 / 0.3

        w = w.float()
        w = w.masked_fill(mask.unsqueeze(-1), -10000.0)
        w = w.masked_fill(t_mask.unsqueeze(1), -10000.0)
        t_w = F.softmax(w, dim=-1)   # B x L x L2

        new_location = t_w.argmax(-1)
        
        if insert_mask:
            
            new_t_w = F.one_hot(new_location, num_classes=new_length).masked_fill(mask.unsqueeze(-1), 0)
            
        else:
            new_t_w = F.one_hot(new_location, num_classes=new_length).masked_fill(mask.unsqueeze(-1), 0)
            new_location = torch.cat((new_location, torch.ones((B, 1)).to(new_location)*new_length), 1)
            new_t_w[(torch.arange(0, new_length, dtype=torch.float32).unsqueeze(0).repeat(B, L, 1).to(new_location) >= new_location[:, :-1].unsqueeze(-1)) &
                    (torch.arange(0, new_length, dtype=torch.float32).unsqueeze(0).repeat(B, L, 1).to(new_location) < new_location[:, 1:].unsqueeze(-1))] = 1
            
        t_x = torch.einsum('bst,bs->bt', new_t_w.to(x).float(), x.float()).long().to(x)
        

        if insert_mask:
            t_x[torch.where(t_x == pad)] = 0
            
        t_x = t_x.masked_fill(t_mask, pad)
        
        return t_x.squeeze()

def dynamic_upsample_token_finalized(x, rate=2, insert_mask=False, insertion_position='uniform'):
    x = x.unsqueeze(0)
    B, L = x.size(0), x.size(1)
    new_length = torch.Tensor([L * rate]).int().item()
    pad = bos = eos = 0
    mask = ~(
        x.ne(pad) & x.ne(bos) & x.ne(eos)
    )

    
    c = torch.arange(1, L+1).unsqueeze(0).repeat(B, 1)
    c = c * rate - (rate/2)
    t = torch.arange(0, new_length).unsqueeze(0).repeat(B, 1)
    # w = -(t[:, None, :] - c[:, :, None]) ** 2 / 0.3
    w = -torch.abs(t[:, None, :] - c[:, :, None])

    w = w.float()
    w = w.masked_fill(mask.unsqueeze(-1), -10000.0)
    # t_w = F.softmax(w, dim=1)   # B x L x L2
    t_w = w   # B x L x L2

    new_location = t_w.argmax(-1)

    if insert_mask:
        
        new_t_w = F.one_hot(new_location, num_classes=new_length).masked_fill(mask.unsqueeze(-1), 0)
        
    else:
        new_t_w = F.one_hot(new_location, num_classes=new_length).masked_fill(mask.unsqueeze(-1), 0)
        new_location = torch.cat((new_location, torch.ones((B, 1)).to(new_location)*new_length), 1)
        new_t_w[(torch.arange(0, new_length, dtype=torch.float32).unsqueeze(0).repeat(B, L, 1).to(new_location) >= new_location[:, :-1].unsqueeze(-1)) &
                (torch.arange(0, new_length, dtype=torch.float32).unsqueeze(0).repeat(B, L, 1).to(new_location) < new_location[:, 1:].unsqueeze(-1))] = 1
        
    t_x = torch.einsum('bst,bs->bt', new_t_w.to(x).float(), x.float()).long().to(x)
    

    if insert_mask:
        t_x[torch.where(t_x == pad)] = 0
        
    # t_x = t_x.masked_fill(t_mask, pad)
    
    return t_x.squeeze()


for length in range(20, 290):
    for rate in np.arange(1.5, 5.5, 0.5):
        x_test = torch.randint(5, 1000, (length,))
        x_test[-10:] = 1

        t_x = dynamic_upsample_token(x_test, rate)
        t_x_pred = dynamic_upsample_token_finalized(x_test, rate)
        
        if not torch.equal(t_x, t_x_pred) :
            print((t_x, t_x_pred))

