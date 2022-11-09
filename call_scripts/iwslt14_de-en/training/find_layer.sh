source $HOME/.bashrc 
conda activate base

bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-H11-UF20M -g 2
bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-H7-UF20M -g 2

bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 1-1-1-1-H11-UF20T -g 2
bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 1-1-1-1-H9-UF20T -g 2
bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 1-1-1-1-H7-UF20T -g 2

bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 1-1-1-1-H11-UF20M -g 2
