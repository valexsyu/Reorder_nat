CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/train_nat.sh -e 2-2-3-1-H12-UF20M --twcc --fp16 --save-interval-updates 70000 --max-update 70000 --max-tokens 3072 -b 12288 -g 4 --dropout 0.1 --no-atten-mask
mkdir checkpoints/2-2-3-1-H12-UF20M/top5_700000steps
cp checkpoints/2-2-3-1-H12-UF20M/checkpoint.best_bleu_*  checkpoints/2-2-3-1-H12-UF20M/top5_700000steps
mkdir checkpoints/2-2-3-1-N-UF20M/
cp checkpoints/2-2-3-1-H12-UF20M/top5_700000steps/* checkpoints/2-2-3-1-N-UF20M/
cp checkpoints/2-2-3-1-H12-UF20M/checkpoint_last.pt checkpoints/2-2-3-1-N-UF20M/
CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/train_nat.sh -e 2-2-3-1-H12-UF20M --twcc --fp16 --save-interval-updates 70000 --max-update 100000 --max-tokens 3072 -b 12288 -g 4 --dropout 0.1 --no-atten-mask
CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/train_nat.sh -e 2-2-3-1-N-UF20M --twcc --fp16 --save-interval-updates 70000 --max-update 100000 --max-tokens 3072 -b 12288 -g 4 --dropout 0.1 --no-atten-mask
