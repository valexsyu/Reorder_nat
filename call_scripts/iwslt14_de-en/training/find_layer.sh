source $HOME/.bashrc 
conda activate base

# bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-H11-UF20M 
# bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-H7-UF20M 

# bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 1-1-1-1-H11-UF20T --max-update 70000 --fp16 --save-interval-updates 70000 --max-tokens 3072
# mkdir checkpoints/1-1-1-1-H9-UF20T
# mkdir checkpoints/1-1-1-1-H7-UF20T
# # cp checkpoints/1-1-1-1-H11-UF20T/checkpoint_last.pt checkpoints/1-1-1-1-H9-UF20T/checkpoint_last.pt
# # cp checkpoints/1-1-1-1-H11-UF20T/checkpoint_last.pt checkpoints/1-1-1-1-H7-UF20T/checkpoint_last.pt
# # cp checkpoints/1-1-1-1-H11-UF20T/checkpoint.best_bleu_*  checkpoints/1-1-1-1-H9-UF20T/
# # cp checkpoints/1-1-1-1-H11-UF20T/checkpoint.best_bleu_*  checkpoints/1-1-1-1-H7-UF20T/
# bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 1-1-1-1-H11-UF20T  --max-update 100000 --fp16 --save-interval-updates 70000 --max-tokens 3072
# bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 1-1-1-1-H9-UF20T  --max-update 100000 --fp16 --save-interval-updates 70000 --max-tokens 3072
# bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 1-1-1-1-H7-UF20T  --max-update 100000 --fp16 --save-interval-updates 70000 --max-tokens 3072

# bash call_scripts/iwslt14_de-en/inference/generate_nat.sh -e 2-2-1-1-H11-UF20M -e 2-2-1-1-H7-UF20M -e 1-1-1-1-H11-UF20T -e 1-1-1-1-H9-UF20T -e 1-1-1-1-H7-UF20T

# bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-H12-UF50M --fp16 --save-interval-updates 70000
# bash call_scripts/iwslt14_de-en/inference/generate_nat.sh -e 2-2-1-1-H12-UF50M
# bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 1-1-1-1-H6-UF20M --fp16 --save-interval-updates 70000
# bash call_scripts/iwslt14_de-en/inference/generate_nat.sh -e 1-1-1-1-H6-UF20M
bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 1-1-1-1-H12-UF20M --fp16 --save-interval-updates 70000
bash call_scripts/iwslt14_de-en/inference/generate_nat.sh -e 1-1-1-1-H12-UF20M
bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 1-1-1-1-H6-UF20M --fp16 --save-interval-updates 70000
bash call_scripts/iwslt14_de-en/inference/generate_nat.sh -e 1-1-1-1-H6-UF20M




