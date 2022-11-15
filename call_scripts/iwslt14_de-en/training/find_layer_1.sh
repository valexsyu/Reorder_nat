source $HOME/.bashrc 
conda activate base

# bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 1-1-1-1-H11-UF20M 
# bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-H12-UR15M --fp16 --save-interval-updates 70000 --max-tokens 3072

# bash call_scripts/iwslt14_de-en/inference/generate_nat.sh -e 1-1-1-1-H11-UF20M -e 2-2-1-1-H12-UR15M

bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-N-UF50M --fp16 --save-interval-updates 70000
bash call_scripts/iwslt14_de-en/inference/generate_nat.sh -e 2-2-1-1-N-UF50M
# bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-H6-UF20T --save-interval-updates 70000
# bash call_scripts/iwslt14_de-en/inference/generate_nat.sh -e 2-2-1-1-H6-UF20T