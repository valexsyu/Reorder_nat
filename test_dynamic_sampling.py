import torch
import torch.nn.functional as F
from fairseq.utils import new_arange, softmax, log_softmax
import pdb 

B, L, D = 5, 10, 26

# # a random tensor x
# x = torch.rand(B, L, D)

# # the (encoder) padding mask
# padding_mask = torch.zeros(B, L)
# padding_mask[:, 90:] = 1
# padding_mask = padding_mask.bool()

# # x = x.transpose(0, 1)  
# mask = padding_mask[..., :L]


# src_upsample = 1.5

   
def dynamic_upsample_tensor(x, mask):
    l = x.new_ones(x.size(0), x.size(1)) * src_upsample
    l = l.masked_fill(mask, 0)
    e = torch.cumsum(l, 1)
    c = e - l / 2
    t = e[:, -1].ceil().long()
    t = new_arange(t, t.max())[None, :].expand(l.size(0), -1)  # B x L2

    t_mask = t >= e[:, -1:]   # target padding mask
    w = -(t[:, None, :] - c[:, :, None]) ** 2 / 0.3
    w = w.float()
    w = w.masked_fill(mask.unsqueeze(-1), -10000.0)
    t_w = F.softmax(w, dim=1)   # B x L x L2
    t_x = torch.einsum('bst,bsd->btd', t_w.type_as(x), x)
    return t_x, t_mask, w, t_w



# a random tensor x
# index 0 is set to padding
x = torch.randint(5, 100, (B, L)).long().cuda()
# x[:, 90:] = 0

# the (encoder) padding mask
padding_mask = torch.zeros(B, L).cuda()
# padding_mask[:, 90:] = 1
padding_mask[:, 9:] = 1
padding_mask = padding_mask.bool()

# x = x.transpose(0, 1)  
mask = padding_mask[..., :L]


src_upsample = 2.5
new_length = int(L * src_upsample)

# modified from
# 
def dynamic_upsample_token(x, mask, insert_mask=False):
    l = x.new_ones(x.size(0), x.size(1)) * src_upsample
    l = l.masked_fill(mask, 0)
    e = torch.cumsum(l, 1)
    c = e - l / 2
    t = e[:, -1].ceil().long()
    # pdb.set_trace()

    # t = new_arange(t, t.max())[None, :].expand(l.size(0), -1)  # B x L2
    t = new_arange(t, new_length)[None, :].expand(l.size(0), -1)  # B x L2

    t_mask = t >= e[:, -1:]   # target padding mask
    w = -(t[:, None, :] - c[:, :, None]) ** 2 / 0.3

    w = w.float()
    w = w.masked_fill(mask.unsqueeze(-1), -10000.0)
    w = w.masked_fill(t_mask.unsqueeze(1), -10000.0)
    t_w = F.softmax(w, dim=1)   # B x L x L2

    new_location = t_w.argmax(-1)
    
    if insert_mask:
        
        new_t_w = F.one_hot(new_location, num_classes=new_length).masked_fill(mask.unsqueeze(-1), 0)
        
    else:
        new_location = torch.cat((new_location, torch.ones((B, 1)).to(new_location)*new_length), 1)
        location_to_exp = 2**(new_location+1)
        diff = (location_to_exp[:, 1:]-location_to_exp[:, :-1])
        diff_to_bit = integer2bit(diff, new_length+1)
        new_t_w = torch.flip(diff_to_bit[:, :, :-1], (2,))
        
    t_x = torch.einsum('bst,bs->bt', new_t_w.to(x).float(), x.float()).int()
    # t_x = torch.matmul(x.float(), new_t_w.to(x).float()).to(x)
    

    if insert_mask:
        t_x[torch.where(t_x == 0)] = 4
        
    t_x = t_x.masked_fill(t_mask, 0)
        
    return t_x, t_mask, w, t_w, new_t_w, new_location


# https://github.com/KarenUllrich/pytorch-binary-converter/blob/master/binary_converter.py
def integer2bit(integer, num_bits=8):
  """Turn integer tensor to binary representation.
      Args:
          integer : torch.Tensor, tensor with integers
          num_bits : Number of bits to specify the precision. Default: 8.
      Returns:
          Tensor: Binary tensor. Adds last dimension to original tensor for
          bits.
  """
  dtype = integer.type()
  exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
  exponent_bits = exponent_bits.repeat(integer.shape + (1,))
  out = integer.unsqueeze(-1) / 2 ** exponent_bits
  return (out - (out % 1)) % 2



print('insert mask:')
t_x, t_mask, w, t_w, new_t_w, new_location = dynamic_upsample_token(x, mask, insert_mask=True)

print('original tokens:', x[0])
print('upsampled tokens:', t_x[0])

print('\ndo not insert mask:')
t_x, t_mask, w, t_w, new_t_w, new_location = dynamic_upsample_token(x, mask, insert_mask=False)

print('original tokens:', x[0])
print('upsampled tokens:', t_x[0])
