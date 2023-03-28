# Contributing to Facebook AI Research Sequence-to-Sequence Toolkit (fairseq)
We want to make contributing to this project as easy and transparent as
possible.

## Train a model

Then we can train a nonautoregressive model using the train_nat.sh script.
* -e : expermiment ID , e.g. m-8-1-1-H12-UF20M . 
  * m: dataset
  * 8: model
  * 1: Frezze parameter
  * 1: Functions 
  * H12: Distill method
  * UF20M: Upsample rate is 2 and use MASK insertion
* --g : GPU number.
* --max-token : 2048 (depend on you gpu memeory , This must be a factor of 12288. e.g. 1024/2048/3072/4096...etc,) 
* --dryrun : do not report to wandb
* --has-eos --wandb-team-id: Keep them.
* --max-update 100000 --lm-start-step 75000 : training steps. Normally is 100k.
* --save-interval-updates : save a checkpoint (and validate) every N updates<




```bash
bash call_scripts/train_nat.sh -e m-8-1-1-H12-UF20M-YOUR-TEST-ID \
                               --save-interval-updates 70000 --max-tokens 2048 \
                               --has-eos --max-update 100000 --lm-start-step 75000 \
                               --wandb-team-id sp-111-2 \
                               -g 1 --fp16
```


## Generation
we can generate by using the generate_nat.sh script.
* -e : expermiment ID , e.g. m-8-1-1-H12-UF20M . 
* -b : batch size s
* --top : To generate the checkpoint_best_top5.pt file, which is averaged the best top 5 files and generate automatically by train_nat.py. 

```bash
CUDA_VISIBLE_DEVICES=0 bash call_scripts/generate_nat.sh -b 20 --local\
                       --ck-types top \
                        -e m-8-1-1-H12-UF20M-YOUR-TEST-ID 
```


## Team setting
* message us your wandb(https://wandb.ai/site) ID.<br>
* create the path to put dataset and pretrained model.
```
  dataroot="../../dataset/nat"
  modelroot="../../dataset/model"
  mkdir -p dataroot modelroot
```
