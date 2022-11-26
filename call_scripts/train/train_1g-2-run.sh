source $HOME/.bashrc 
conda activate base
bash call_scripts/train_nat.sh -e J-6-1-1-N-UF30T --fp16 --save-interval-updates 70000 --max-tokens 2048 --no-atten-mask
bash call_scripts/train_nat.sh -e 2-2-1-1-H12-UF50T --fp16 --save-interval-updates 70000 --max-tokens 2048 --no-atten-mask

