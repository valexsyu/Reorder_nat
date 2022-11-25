source $HOME/.bashrc 
conda activate base
bash call_scripts/train_nat.sh -e K-2-1-1-H12-UR40M --fp16 --save-interval-updates 70000 --max-tokens 2048 --no-atten-mask
bash call_scripts/generate_nat.sh -e K-2-1-1-H12-UR40M -b 1 --data-subset test-valid --no-atten-mask