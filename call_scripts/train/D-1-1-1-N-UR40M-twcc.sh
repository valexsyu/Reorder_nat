CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/train_nat.sh -e D-1-1-1-N-UR40M \
    --max-update 20000 --save-interval-updates 10000 --max-tokens 2048 -b 65536 --fp16 \
    --no-atten-mask \
    --warmup-updates 3000 \
    --lm-start-step 20000 \
    --dropout 0.1 \
    --twcc \
    --watch-test-bleu \
    -g 4

mkdir checkpoints/D-1-1-1-N-UR40M/top5_20000steps
cp checkpoints/D-1-1-1-N-UR40M/checkpoint.best_bleu_*  checkpoints/D-1-1-1-N-UR40M/top5_20000steps
mkdir checkpoints/D-1-1-1-H12-UR40M/
cp checkpoints/D-1-1-1-N-UR40M/top5_20000steps/* checkpoints/D-1-1-1-H12-UR40M/
cp checkpoints/D-1-1-1-N-UR40M/checkpoint_last.pt checkpoints/D-1-1-1-H12-UR40M/checkpoint_last.pt


CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/train_nat.sh -e D-1-1-1-N-UR40M \
    --max-update 30000 --save-interval-updates 10000 --max-tokens 2048 -b 65536 --fp16 \
    --no-atten-mask \
    --warmup-updates 3000 \
    --lm-start-step 20000 \
    --dropout 0.1 \
    --twcc \
    --watch-test-bleu \
    -g 4

CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/train_nat.sh -e D-1-1-1-H12-UR40M \
    --max-update 30000 --save-interval-updates 10000 --max-tokens 2048 -b 65536 --fp16 \
    --no-atten-mask \
    --warmup-updates 3000 \
    --lm-start-step 20100 \
    --dropout 0.1 \
    --twcc \
    --watch-test-bleu \
    -g 4

bash call_scripts/generate_nat.sh -b 1 --data-subset test-valid --no-atten-mask --twcc \
-e D-1-1-1-N-UR40M

bash call_scripts/generate_nat.sh -b 1 --data-subset test-valid --no-atten-mask --twcc \
-e D-1-1-1-H12-UR40M