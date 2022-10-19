source $HOME/.bashrc 
conda activate base
echo "============================================================================================"
echo "                   training              No-7-2-30                                           "
echo "============================================================================================"
bash call_scripts/iwslt14_de-en/training/No-7-2-30-translation_mask.sh
echo "============================================================================================"
echo "                   training              No-7-2-33                                             "
echo "============================================================================================"
mkdir checkpoints/No-7-2-33-translation-lm_mask
cp checkpoints/No-7-2-30-translation_mask/checkpoint_*_70000.pt checkpoints/No-7-2-33-translation-lm_mask/checkpoint_last.pt
bash call_scripts/iwslt14_de-en/training/No-7-2-33-translation-lm_mask.sh
