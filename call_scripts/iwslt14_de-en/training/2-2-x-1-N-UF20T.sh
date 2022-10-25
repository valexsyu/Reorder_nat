source $HOME/.bashrc 
conda activate base

mkdir checkpoints/2-2-2-1-N-UF20T
cp checkpoints/2-2-2-1-H12-UF20T/checkpoint_*_70000.pt checkpoints/2-2-2-1-N-UF20T/checkpoint_last.pt
bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-2-1-N-UF20T

mkdir checkpoints/2-2-3-1-N-UF20T
cp checkpoints/2-2-3-1-H12-UF20T/checkpoint_*_70000.pt checkpoints/2-2-3-1-N-UF20T/checkpoint_last.pt
bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-3-1-N-UF20T

mkdir checkpoints/2-2-4-1-N-UF20T
cp checkpoints/2-2-4-1-H12-UF20T/checkpoint_*_70000.pt checkpoints/2-2-4-1-N-UF20T/checkpoint_last.pt
bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-4-1-N-UF20T

