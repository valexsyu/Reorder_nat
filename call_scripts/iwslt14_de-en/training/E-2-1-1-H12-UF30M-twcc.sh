CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/iwslt14_de-en/training/train_nat.sh -e E-2-1-1-H12-UF30M --twcc --fp16 --save-interval-updates 70000 --max-update 100000 --max-tokens 2048 -b 65536 -g 8 --dropout 0.2