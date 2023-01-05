experiment_1=2-2-4-1-N-UF20T
CUDA_VISIBLE_DEVICES=0 bash call_scripts/train_nat.sh -e $experiment_1 --local --fp16 --save-interval-updates 70000 --max-update 100000 --max-tokens 1024 -b 12288 -g 1 --dropout 0.1 --no-atten-mask
