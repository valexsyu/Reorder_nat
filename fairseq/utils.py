# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import os
import torch
from torch.autograd import Variable
from torch.serialization import default_restore_location

from fairseq import criterions, data, models


def build_model(args, dataset):
    if args.arch == 'fconv':
        encoder_layers = eval(args.encoder_layers)
        decoder_layers = eval(args.decoder_layers)
        decoder_attention = eval(args.decoder_attention)
        model = models.fconv(
            dataset, args.dropout, args.encoder_embed_dim, encoder_layers,
            args.decoder_embed_dim, decoder_layers, decoder_attention,
            decoder_out_embed_dim=args.decoder_out_embed_dim)
    else:
        model = models.__dict__[args.arch](dataset, args.dropout)
    return model


def build_criterion(args, dataset):
    padding_idx = dataset.dst_dict.pad()
    if args.label_smoothing > 0:
        return criterions.LabelSmoothedCrossEntropyCriterion(args.label_smoothing, padding_idx)
    else:
        return criterions.CrossEntropyCriterion(padding_idx)


def torch_persistent_save(*args, **kwargs):
    for i in range(3):
        try:
            return torch.save(*args, **kwargs)
        except:
            if i == 3:
                raise


def save_checkpoint(args, epoch, batch_offset, model, optimizer, lr_scheduler, val_loss=None):
    state_dict = {
        'args': args,
        'epoch': epoch,
        'batch_offset': batch_offset,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_loss': lr_scheduler.best,
        'val_loss': val_loss,
    }

    if batch_offset == 0:
        if not args.no_epoch_checkpoints:
            epoch_filename = os.path.join(args.save_dir, 'checkpoint{}.pt'.format(epoch))
            torch_persistent_save(state_dict, epoch_filename)

        assert val_loss is not None
        if not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best:
            save_checkpoint.best = val_loss
            best_filename = os.path.join(args.save_dir, 'checkpoint_best.pt')
            torch_persistent_save(state_dict, best_filename)

    last_filename = os.path.join(args.save_dir, 'checkpoint_last.pt')
    torch_persistent_save(state_dict, last_filename)


def load_checkpoint(filename, model, optimizer, lr_scheduler, cuda_device=None):
    if not os.path.exists(filename):
        return 1, 0
    if cuda_device is None:
        state = torch.load(filename)
    else:
        state = torch.load(filename, map_location=lambda s,l:
            default_restore_location(s, 'cuda:{}'.format(cuda_device)))

    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    lr_scheduler.best = state['best_loss']
    epoch = state['epoch'] + 1
    batch_offset = state['batch_offset']

    gpu_str = ' on GPU #{}'.format(cuda_device) if cuda_device is not None else ''
    print('| loaded checkpoint {} (epoch {}){}'.format(filename, epoch, gpu_str))
    return epoch, batch_offset


def load_ensemble_for_inference(models, data_path):
    # load model architectures and weights
    states = []
    for model in models:
        if not os.path.exists(model):
            raise IOError('Model file not found: ' + model)
        states.append(
            torch.load(model, map_location=lambda s,l: default_restore_location(s, 'cpu'))
        )

    # load dataset
    args = states[0]['args']
    dataset = data.load(data_path, args.source_lang, args.target_lang)

    # build models
    models = []
    for state in states:
        model = build_model(args, dataset)
        model.load_state_dict(state['model'])
        models.append(model)

    return models, dataset


def prepare_sample(sample, volatile=False, cuda_device=None):
    """Wrap input tensors in Variable class."""
    r = {
        'id': sample['id'],
        'ntokens': sample['ntokens'],
        'net_input': {},
    }
    # prepare input to network
    for key in ['src_tokens', 'src_positions', 'input_tokens', 'input_positions', 'target']:
        tensor = sample[key]
        if cuda_device is not None and torch.cuda.is_available():
            tensor = tensor.cuda(async=True, device=cuda_device)
        r['net_input'][key] = Variable(tensor, volatile=volatile)
    return r
