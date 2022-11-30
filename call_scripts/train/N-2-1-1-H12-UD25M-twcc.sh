CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/train_nat.sh -e N-2-1-1-H12-UD25M --twcc \
    --fp16 -g 4 --save-interval-updates 32500 --max-update 200000 --lm-start-step 130000 \
    --max-tokens 4095 -b 65520 --no-atten-mask


bash call_scripts/generate_nat.sh -b 1 --data-subset test --no-atten-mask --twcc --ch-types top \
    -e N-2-1-1-H12-UD25M