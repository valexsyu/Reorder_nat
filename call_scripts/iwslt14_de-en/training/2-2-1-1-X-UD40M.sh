source $HOME/.bashrc 
conda activate base

bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-N-UD40M -g 5 --max-tokens 1228 --max-update 70000
mkdir checkpoints/2-2-1-1-H12-UD40M
cp checkpoints/2-2-1-1-N-UD40M/checkpoint_last.pt checkpoints/2-2-1-1-H12-UD40M/checkpoint_last.pt
cp checkpoints/2-2-1-1-N-UD40M/checkpoint.best_bleu_*  checkpoints/2-2-1-1-H12-UD40M/
bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-N-UD40M -g 5 --max-tokens 1228 --max-update 100000
bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-H12-UD40M -g 5 --max-tokens 1228 --max-update 100000