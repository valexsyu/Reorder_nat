source $HOME/.bashrc 
conda activate base

bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-H7-UR40M --fp16 -g 4 --save-interval-updates 70000 --max-tokens 1536
bash call_scripts/iwslt14_de-en/inference/generate_nat.sh -e 2-2-1-1-H7-UR40M
# bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-H7-UR40M --fp16 -g 4 --save-interval-updates 70000 --max-tokens 1536 --dropout 0.3
# bash call_scripts/iwslt14_de-en/inference/generate_nat.sh -e 2-2-1-1-H7-UR40M
bash call_scripts/iwslt14_de-en/training/train_nat.sh -e F-2-1-1-N-UR40M --fp16 -g 4 --save-interval-updates 70000 --max-tokens 2048 -b 65536
bash call_scripts/iwslt14_de-en/inference/generate_nat.sh -e F-2-1-1-N-UR40M 