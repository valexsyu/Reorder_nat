source $HOME/.bashrc 
conda activate base
echo "============================================================================================"
echo "                   training              No-6-4-04                                           "
echo "============================================================================================"
bash call_scripts/iwslt14_de-en/training/No-6-4-04-translation-valid.sh
echo "============================================================================================"
echo "                   training              No-6-4-05                                             "
echo "============================================================================================"
mkdir checkpoints/No-6-4-05-translation-lm-valid
cp checkpoints/No-6-4-04-translation-valid/checkpoint_*_70000.pt checkpoints/No-6-4-05-translation-lm-valid/checkpoint_last.pt
bash call_scripts/iwslt14_de-en/training/No-6-4-05-translation-lm-valid.sh
