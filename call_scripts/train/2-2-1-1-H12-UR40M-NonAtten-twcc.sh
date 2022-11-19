CUDA_VISIBLE_DEVICES=0 bash call_scripts/train_nat.sh -e 2-2-1-1-H12-UR40M --twcc --fp16 \
                            --save-interval-updates 70000 --max-update 100000 --max-tokens 1536 -b 12288 -g 2 --dropout 0.1
CUDA_VISIBLE_DEVICES=0 bash call_scripts/iwslt14_de-en/inference/generate_nat.sh --twcc -b 60 -e 2-2-1-1-H12-UR40M