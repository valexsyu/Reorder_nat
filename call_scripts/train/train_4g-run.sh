source $HOME/.bashrc 
conda activate base
# bash call_scripts/train_nat.sh -e E-2-1-1-H12-UR40M --fp16 -g 4 --save-interval-updates 32500 --max-update 130000 --lm-start-step 130000 --max-tokens 2048 -b 65536 --no-atten-mask
# mkdir checkpoints/E-2-1-1-H12-UR40M/top5_130000steps
# cp checkpoints/E-2-1-1-H12-UR40M/checkpoint.best_bleu_*  checkpoints/E-2-1-1-H12-UR40M/top5_700000steps
# mkdir checkpoints/E-2-1-1-N-UR40M/
# cp checkpoints/E-2-1-1-H12-UR40M/top5_700000steps/* checkpoints/E-2-1-1-N-UR40M/
# cp checkpoints/E-2-1-1-H12-UR40M/checkpoint_last.pt checkpoints/E-2-1-1-N-UR40M/checkpoint_last.pt
# bash call_scripts/train_nat.sh -e E-2-1-1-H12-UR40M --fp16 -g 4 --save-interval-updates 32500 --max-update 200000 --lm-start-step 130000 --max-tokens 2048 -b 65536 --no-atten-mask
bash call_scripts/generate_nat.sh -b 1 --data-subset test --no-atten-mask \
-e E-2-1-1-H12-UR40M \

bash call_scripts/train_nat.sh -e E-2-1-1-H4-UF40M --fp16 -g 4 --save-interval-updates 10000 --max-update 130000 --lm-start-step 130000 --max-tokens 2048 -b 65536 --no-atten-mask
