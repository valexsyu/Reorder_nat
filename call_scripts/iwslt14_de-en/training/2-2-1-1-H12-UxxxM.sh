source $HOME/.bashrc 
conda activate base

bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-H12-UF40M -g 3 --max-tokens 1024
bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-H12-UR45M -g 3
bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-H12-UR50M -g 3