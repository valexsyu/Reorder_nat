source $HOME/.bashrc 
conda activate base

mkdir checkpoints/1-1-2-1-N-UF20T
cp checkpoints/1-1-2-1-H12-UF20T/checkpoint_*_70000.pt checkpoints/1-1-2-1-N-UF20T/checkpoint_last.pt
bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 1-1-2-1-N-UF20T

mkdir checkpoints/1-1-3-1-N-UF20T
cp checkpoints/1-1-3-1-H12-UF20T/checkpoint_*_70000.pt checkpoints/1-1-3-1-N-UF20T/checkpoint_last.pt
bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 1-1-3-1-N-UF20T

mkdir checkpoints/1-1-4-1-N-UF20T
cp checkpoints/1-1-4-1-H12-UF20T/checkpoint_*_70000.pt checkpoints/1-1-4-1-N-UF20T/checkpoint_last.pt
bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 1-1-4-1-N-UF20T

