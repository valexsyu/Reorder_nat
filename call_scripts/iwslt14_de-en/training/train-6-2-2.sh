source $HOME/.bashrc 
conda activate base
echo "============================================================================================"
echo "                   training              No-6-2-20                                             "
echo "============================================================================================"
bash call_scripts/iwslt14_de-en/training/No-6-2-20-translation.sh
echo "============================================================================================"
echo "                   training              No-6-2-23                                             "
echo "============================================================================================"
mkdir checkpoints/No-6-2-23-translation-lm 
cp checkpoints/No-6-2-20-translation/checkpoint_*_70000.pt checkpoints/No-6-2-23-translation-lm/checkpoint_last.pt
bash call_scripts/iwslt14_de-en/training/No-6-2-23-translation-lm.sh
