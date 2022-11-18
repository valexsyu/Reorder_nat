CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/iwslt14_de-en/training/train_nat.sh \
-e F-2-1-1-H12-UR40M --twcc --fp16 --save-interval-updates 100000 --max-update 300000 --lm-start-step 200000 \
--max-tokens 2048 -b 65536 -g 4 --dropout 0.1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/iwslt14_de-en/inference/generate_nat.sh --twcc -b 60 \
-e F-2-1-1-H12-UR40M