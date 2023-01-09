experiment_1=1-1-4-1-N-UF20T-new
CUDA_VISIBLE_DEVICES=0 bash call_scripts/train_nat.sh -e $experiment_1 --local --fp16 --save-interval-updates 70000 --max-update 100000 --max-tokens 1024 -b 12288 -g 1 --dropout 0.1 --no-atten-mask
CUDA_VISIBLE_DEVICES=0 bash call_scripts/generate_nat.sh -b 1 --local --data-subset test --ck-types top --avg-speed 1 --no-atten-mask --avg-ck-turnoff --skip-exist-genfile --load-exist-bleu \
    -e $experiment_1 \


