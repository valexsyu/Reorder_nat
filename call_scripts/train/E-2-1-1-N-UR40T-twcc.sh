CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/train_nat.sh -e E-2-1-1-N-UR40T --twcc --fp16 -g 4 --save-interval-updates 32500 --max-update 200000 --lm-start-step 130000 --max-tokens 2730 -b 65520 --no-atten-mask