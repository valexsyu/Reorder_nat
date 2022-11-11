source $HOME/.bashrc 
conda activate base

bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 1-1-1-1-H11-UF20M 
bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-H12-UR15M --fp16 --save-interval-updates 70000 --max-tokens 3072