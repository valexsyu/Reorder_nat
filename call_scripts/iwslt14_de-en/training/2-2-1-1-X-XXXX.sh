source $HOME/.bashrc 
conda activate base

bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-N-UF50T --fp16 --save-interval-updates 70000
