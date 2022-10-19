source $HOME/.bashrc 
conda activate base
echo "============================================================================================"
echo "                   training              No-6-2-34                                            "
echo "============================================================================================"
bash call_scripts/iwslt14_de-en/training/No-6-2-34-translation_mask.sh
echo "============================================================================================"
echo "                   training              No-6-2-35                                             "
echo "============================================================================================"
mkdir checkpoints/No-6-2-35-translation-lm_mask
cp checkpoints/No-6-2-34-translation_mask/checkpoint_*_70000.pt checkpoints/No-6-2-35-translation-lm_mask/checkpoint_last.pt
bash call_scripts/iwslt14_de-en/training/No-6-2-35-translation-lm_mask.sh
