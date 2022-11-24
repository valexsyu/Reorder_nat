CUDA_VISIBLE_DEVICES=0 bash call_scripts/train_nat.sh -e 2-2-2-1-N-UF20T --twcc --fp16 --save-interval-updates 70000 --max-update 70000 --max-tokens 4096 -b 12288 -g 1 --dropout 0.1 --no-atten-mask
mkdir checkpoints/2-2-2-1-N-UF20T/top5_700000steps
cp checkpoints/2-2-2-1-N-UF20T/checkpoint.best_bleu_*  checkpoints/2-2-2-1-N-UF20T/top5_700000steps
mkdir checkpoints/2-2-2-1-H12-UF20T/
cp checkpoints/2-2-2-1-N-UF20T/top5_700000steps/* checkpoints/2-2-2-1-H12-UF20T/
cp checkpoints/2-2-2-1-N-UF20T/checkpoint_last.pt checkpoints/2-2-2-1-H12-UF20T/checkpoint_last.pt
CUDA_VISIBLE_DEVICES=0 bash call_scripts/train_nat.sh -e 2-2-2-1-N-UF20T --twcc --fp16 --save-interval-updates 70000 --max-update 100000 --max-tokens 4096 -b 12288 -g 1 --dropout 0.1 --no-atten-mask
CUDA_VISIBLE_DEVICES=0 bash call_scripts/train_nat.sh -e 2-2-2-1-H12-UF20T --twcc --fp16 --save-interval-updates 70000 --max-update 100000 --max-tokens 4096 -b 12288 -g 1 --dropout 0.1 --no-atten-mask
CUDA_VISIBLE_DEVICES=0 bash call_scripts/generate_nat.sh --twcc -b b0 --data-subset test-valid --no-atten-mask \
-e 2-2-2-1-N-UF20T -e 2-2-2-1-H12-UF20T