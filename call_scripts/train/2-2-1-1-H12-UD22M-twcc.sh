CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/train_nat.sh -e 2-2-1-1-H12-UD22M --twcc --save-interval-updates 70000 \
--max-update 100000 --max-tokens 3072 -b 12288 -g 4 --dropout 0.1 --no-atten-mask