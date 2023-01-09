bash call_scripts/train_nat.sh -e  e-7-1-1-H12-UR40M --fp16 -g 1 \
                               --save-interval-updates 70000 --max-update 100000 --lm-start-step 75000 \
                               --max-tokens 2048 -b 65536 --no-atten-mask --dryrun --valid-set --cpu 