import torch
import torch.nn.functional as F
from fairseq.utils import new_arange, softmax, log_softmax

B, L= 5, 10

x = torch.randint(5, 100, (B, L))
x[:, 90:] = 0

# the (encoder) padding mask
padding_mask = torch.zeros(B, L)
padding_mask[:, 90:] = 1
padding_mask = padding_mask.bool()

# x = x.transpose(0, 1)  
mask = padding_mask[..., :L]
src_upsample=1.5

   
# def dynamic_upsample(x, mask):
#     l = x.new_ones(x.size(1), x.size(0)) * src_upsample
#     l = l.masked_fill(mask, 0)
#     e = torch.cumsum(l, 1)
#     c = e - l / 2
#     t = e[:, -1].ceil().long()
#     t = new_arange(t, t.max())[None, :].expand(l.size(0), -1)  # B x L2
#     t_mask = t >= e[:, -1:]   # target padding mask
#     w = -(t[:, None, :] - c[:, :, None]) ** 2 / 0.3
#     w = w.float()
#     w = w.masked_fill(mask.unsqueeze(-1), -10000.0)
#     t_w = F.softmax(w, dim=1)   # B x L x L2
#     t_x = torch.einsum('ik,kj->ij', t_w.type_as(x), x)
#     return t_x, t_mask, w


# t_x, t_mask, w = dynamic_upsample(x, mask)


# rate = src_upsample
# source = x
# def _random_mask(source, rate, source_masks):
    # pad = self.src_dict.pad()
    # bos = self.src_dict.bos()
    # eos = self.src_dict.eos()
    # unk = self.src_dict.unk()

    # source_masks = (
    #     source.ne(pad) & source.ne(bos) & source.ne(eos)
    # )
#     import pdb;pdb.set_trace()
#     source_score = source.clone().float().uniform_()
#     source_score.masked_fill_(~source_masks, 2.0)
#     source_length = source_masks.sum(1).float()*(rate-1)
#     source_length = source_length * source_length.clone().uniform_()
#     source_length = source_length + 1  # make sure to mask at least one token.

#     _, source_rank = source_score.sort(1)
#     source_cutoff = new_arange(source_rank) < source_length[:, None].long()
#     output = source.masked_fill(
#         source_cutoff.scatter(1, source_rank, source_cutoff), 4
#     )
#     return output
# print(x)
# output = _random_mask(source, rate, mask)
# print(output)

# def dynamic_upsample(source, masks, rate):
#     import pdb;pdb.set_trace()
#     pad_index = 0
#     mask_index =4
    
#     upsampled_length = torch.ceil(torch.cumsum(mask, 1)*rate).long()
#     upsample_source = torch.full((source.size(0), source.size(1)), pad_index)
#     upsample_source[:, [l for l in source_length]] = mask_index
    
#     return upsample_source

# rate = src_upsample
# source = x
# output = dynamic_upsample(source, mask, rate )


def dynamic_upsample_token(x, mask, rate):
    pad_index = 0
    mask_index =4
    new_length = int(x.size(1) * rate)
    l = x.new_ones(x.size(0), x.size(1)) * rate
    l = l.masked_fill(mask, 0)
    e = torch.cumsum(l, 1)
    c = e - l / 2
    t = e[:, -1].ceil().long()
    import pdb; pdb.set_trace()

    # t = new_arange(t, t.max())[None, :].expand(l.size(0), -1)  # B x L2
    t = new_arange(t, new_length)[None, :].expand(l.size(0), -1)  # B x L2

    t_mask = t >= e[:, -1:]   # target padding mask
    w = -(t[:, None, :] - c[:, :, None]) ** 2 / 0.3

    w = w.float()
    w = w.masked_fill(mask.unsqueeze(-1), -10000.0)
    w = w.masked_fill(t_mask.unsqueeze(1), -10000.0)
    t_w = F.softmax(w, dim=1)   # B x L x L2

    new_location = t_w.argmax(-1)
    new_t_w = F.one_hot(new_location, num_classes=new_length).masked_fill(mask.unsqueeze(-1), 0)
    t_x = torch.einsum('bst,bs->bt', new_t_w.to(x).float(), x.float()).int()

    t_x[torch.where(t_x == 0)] = mask_index
    t_x = t_x.masked_fill(t_mask, 0)
    return t_x, t_mask, w, t_w, new_t_w, new_location


rate = src_upsample
source = x
# output = dynamic_upsample(source, mask, rate )
t_x, t_mask, w, t_w, new_t_w, new_location = dynamic_upsample_token(source, mask , rate)



print('original tokens:', x[0])
print('upsampled tokens:', t_x[0])
