source $HOME/.bashrc 
conda activate base

bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-H11-UF20M -g 2
bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-H7-UF20M -g 2

bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 1-1-1-1-H11-UF20T -g 2 --max-update 70000
mkdir checkpoints/1-1-1-1-H9-UF20T
mkdir checkpoints/1-1-1-1-H7-UF20T
cp checkpoints/1-1-1-1-H11-UF20T/checkpoint_last.pt checkpoints/1-1-1-1-H9-UF20T/checkpoint_last.pt
cp checkpoints/1-1-1-1-H11-UF20T/checkpoint_last.pt checkpoints/1-1-1-1-H7-UF20T/checkpoint_last.pt
cp checkpoints/1-1-1-1-H11-UF20T/checkpoint.best_bleu_*  checkpoints/1-1-1-1-H9-UF20T/
cp checkpoints/1-1-1-1-H11-UF20T/checkpoint.best_bleu_*  checkpoints/1-1-1-1-H7-UF20T/
bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 1-1-1-1-H11-UF20T -g 2 --max-update 100000
bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 1-1-1-1-H9-UF20T -g 2 --max-update 100000
bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 1-1-1-1-H7-UF20T -g 2 --max-update 100000

bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 1-1-1-1-H11-UF20M -g 2
