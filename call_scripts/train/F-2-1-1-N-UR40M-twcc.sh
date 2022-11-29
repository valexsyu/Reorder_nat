# CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/train_nat.sh -e F-2-1-1-N-UR40M --twcc --fp16 -g 4 --save-interval-updates 32500 --max-update 130000 --lm-start-step 130000 --max-tokens 2730 -b 65520 --no-atten-mask
# mkdir checkpoints/F-2-1-1-N-UR40M/top5_130000steps
# cp checkpoints/F-2-1-1-N-UR40M/checkpoint.best_bleu_*  checkpoints/F-2-1-1-N-UR40M/top5_130000steps
# mkdir checkpoints/F-2-1-1-H12-UR40M/
# cp checkpoints/F-2-1-1-N-UR40M/top5_130000steps/* checkpoints/F-2-1-1-H12-UR40M/
# cp checkpoints/F-2-1-1-N-UR40M/checkpoint_last.pt checkpoints/F-2-1-1-H12-UR40M/checkpoint_last.pt
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/train_nat.sh -e F-2-1-1-N-UR40M --twcc --fp16 -g 4 --save-interval-updates 32500 --max-update 200000 --lm-start-step 130000 --max-tokens 2730 -b 65520 --no-atten-mask
# bash call_scripts/generate_nat.sh -b 50 --data-subset test-valid --no-atten-mask --twcc \
# -e F-2-1-1-N-UR40M

CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/train_nat.sh -e F-2-1-1-N-UR40M \
    --twcc --fp16 -g 4 --save-interval-updates 32500 --max-update 200000 \
    --lm-start-step 130000 --max-tokens 1638 -b 65520 \
    --no-atten-mask


bash call_scripts/generate_nat.sh -b 1 --data-subset test-valid --no-atten-mask --twcc \
-e F-2-1-1-N-UR40M