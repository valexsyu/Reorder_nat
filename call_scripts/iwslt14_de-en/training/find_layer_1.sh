source $HOME/.bashrc 
conda activate base
# bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-H11-UF20T -g 2
# bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-H9-UF20T -g 2
# bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-H7-UF20T -g 2

# mkdir checkpoints/2-2-1-1-H11-UF20M
# cp checkpoints/2-2-1-1-N-UF20M/checkpoint_*_70000.pt checkpoints/2-2-1-1-H11-UF20M/checkpoint_last.pt
# mkdir checkpoints/2-2-1-1-H9-UF20M
# cp checkpoints/2-2-1-1-N-UF20M/checkpoint_*_70000.pt checkpoints/2-2-1-1-H9-UF20M/checkpoint_last.pt
# mkdir checkpoints/2-2-1-1-H7-UF20M
# cp checkpoints/2-2-1-1-N-UF20M/checkpoint_*_70000.pt checkpoints/2-2-1-1-H7-UF20M/checkpoint_last.pt

# bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-H11-UF20M -g 1
# bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-H9-UF20M -g 1
# bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-H7-UF20M -g 1

# mkdir checkpoints/1-1-1-1-H11-UF20T
# cp checkpoints/No-7-4-00-translation/checkpoint_*_70000.pt checkpoints/1-1-1-1-H11-UF20T/checkpoint_last.pt
# mkdir checkpoints/1-1-1-1-H9-UF20T
# cp checkpoints/No-7-4-00-translation/checkpoint_*_70000.pt checkpoints/1-1-1-1-H9-UF20T/checkpoint_last.pt
# mkdir checkpoints/1-1-1-1-H7-UF20T
# cp checkpoints/No-7-4-00-translation/checkpoint_*_70000.pt checkpoints/1-1-1-1-H7-UF20T/checkpoint_last.pt


# bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 1-1-1-1-H11-UF20T -g 1
# bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 1-1-1-1-H9-UF20T -g 1
# bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 1-1-1-1-H7-UF20T -g 1

mkdir checkpoints/1-1-1-1-H11-UF20M
cp checkpoints/1-1-1-1-N-UF20M/checkpoint_*_70000.pt checkpoints/1-1-1-1-H11-UF20M/checkpoint_last.pt
mkdir checkpoints/1-1-1-1-H9-UF20M
cp checkpoints/1-1-1-1-N-UF20M/checkpoint_*_70000.pt checkpoints/1-1-1-1-H9-UF20M/checkpoint_last.pt
mkdir checkpoints/1-1-1-1-H7-UF20M
cp checkpoints/1-1-1-1-N-UF20M/checkpoint_*_70000.pt checkpoints/1-1-1-1-H7-UF20M/checkpoint_last.pt


bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 1-1-1-1-H11-UF20M -g 1
bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 1-1-1-1-H9-UF20M -g 1
bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 1-1-1-1-H7-UF20M -g 1