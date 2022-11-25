source $HOME/.bashrc 
conda activate base
# bash call_scripts/train_nat.sh -e E-2-1-1-H12-UF30M --fp16 -g 4 --save-interval-updates 100000 --max-update 200000 --lm-start-step 130000 --max-tokens 2048 -b 65536 --no-atten-mask
# bash call_scripts/generate_nat.sh -e E-2-1-1-H12-UF30M -b 1 --data-subset test-valid --no-atten-mask
bash call_scripts/train_nat.sh -e E-2-1-1-H12-UR40M --fp16 -g 4 --save-interval-updates 32500 --max-update 200000 --lm-start-step 130000 --max-tokens 2048 -b 65536 --no-atten-mask
bash call_scripts/generate_nat.sh -e E-2-1-1-H12-UR40M -b 1 --data-subset test-valid --no-atten-mask

