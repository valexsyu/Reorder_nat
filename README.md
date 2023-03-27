# Contributing to Facebook AI Research Sequence-to-Sequence Toolkit (fairseq)
We want to make contributing to this project as easy and transparent as
possible.

## Train a model

Then we can train a nonautoregressive model using the train_nat.sh script.
-e : expermiment ID , e.g. m-8-1-1-H12-UF20M . <br>
  m: dataset<br>
  8: model<br>
  1: Frezze parameter<br>
  1: Functional<br>
  H12: Distill method<br>
  UF20M: Upsample rate is 2 and use MASK insertion<br>
--g : GPU number.<br>
--max-token : 2048 (depend on you gpu memeory , This must be a factor of 12288. e.g. 1024/2048/3072/4096...etc,) <br>
--dryrun : do not report to wandb<br>
--has-eos --max-update 100000 --lm-start-step 75000: keep them.<br>
--save-interval-updates : save a checkpoint (and validate) every N updates<br>




```bash
bash call_scripts/train_nat.sh -e m-8-1-1-H12-UF20M-YOUR-TEST-ID \
                               --save-interval-updates 70000 --max-tokens 2048 \
                               --has-eos --max-update 100000 --lm-start-step 75000 \
                               --g 1 --fp16
```