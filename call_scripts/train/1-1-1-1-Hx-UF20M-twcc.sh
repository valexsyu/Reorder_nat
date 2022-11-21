CUDA_VISIBLE_DEVICES=0 bash call_scripts/train_nat.sh -e 1-1-1-1-H4-UF20M --twcc --fp16 --save-interval-updates 70000 --max-update 70000 --max-tokens 4096 -b 12288 -g 1 --dropout 0.1 #add attenation
mkdir checkpoints/1-1-1-1-H4-UF20M/top5_700000steps
cp checkpoints/1-1-1-1-H4-UF20M/checkpoint.best_bleu_*  checkpoints/1-1-1-1-H4-UF20M/top5_700000steps
mkdir checkpoints/1-1-1-1-H3-UF20M/
mkdir checkpoints/1-1-1-1-H2-UF20M/
mkdir checkpoints/1-1-1-1-H1-UF20M/
cp checkpoints/1-1-1-1-H4-UF20M/top5_700000steps/* checkpoints/1-1-1-1-H3-UF20M/
cp checkpoints/1-1-1-1-H4-UF20M/top5_700000steps/* checkpoints/1-1-1-1-H2-UF20M/
cp checkpoints/1-1-1-1-H4-UF20M/top5_700000steps/* checkpoints/1-1-1-1-H1-UF20M/
cp checkpoints/1-1-1-1-H4-UF20M/checkpoint_last.pt checkpoints/1-1-1-1-H3-UF20M/checkpoint_last.pt
cp checkpoints/1-1-1-1-H4-UF20M/checkpoint_last.pt checkpoints/1-1-1-1-H2-UF20M/checkpoint_last.pt
cp checkpoints/1-1-1-1-H4-UF20M/checkpoint_last.pt checkpoints/1-1-1-1-H1-UF20M/checkpoint_last.pt
CUDA_VISIBLE_DEVICES=0 bash call_scripts/train_nat.sh -e 1-1-1-1-H4-UF20M --twcc --fp16 --save-interval-updates 70000 --max-update 100000 --max-tokens 4096 -b 12288 -g 1 --dropout 0.1 
CUDA_VISIBLE_DEVICES=0 bash call_scripts/train_nat.sh -e 1-1-1-1-H3-UF20M --twcc --fp16 --save-interval-updates 70000 --max-update 100000 --max-tokens 4096 -b 12288 -g 1 --dropout 0.1
CUDA_VISIBLE_DEVICES=0 bash call_scripts/train_nat.sh -e 1-1-1-1-H2-UF20M --twcc --fp16 --save-interval-updates 70000 --max-update 100000 --max-tokens 4096 -b 12288 -g 1 --dropout 0.1
CUDA_VISIBLE_DEVICES=0 bash call_scripts/train_nat.sh -e 1-1-1-1-H1-UF20M --twcc --fp16 --save-interval-updates 70000 --max-update 100000 --max-tokens 4096 -b 12288 -g 1 --dropout 0.1
CUDA_VISIBLE_DEVICES=0 bash call_scripts/generate_nat.sh --twcc -b 60 --data-subset test-valid \
-e 1-1-1-1-H5-UF20M -e 1-1-1-1-H4-UF20M -e 1-1-1-1-H3-UF20M -e 1-1-1-1-H2-UF20M -e 1-1-1-1-H1-UF20M