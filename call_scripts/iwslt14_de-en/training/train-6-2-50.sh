source $HOME/.bashrc 
conda activate base
echo "============================================================================================"
echo "                   training              No-6-2-50                                            "
echo "============================================================================================"
bash call_scripts/iwslt14_de-en/training/No-6-2-50-translation-dy.sh
echo "============================================================================================"
echo "                   training              No-6-2-53                                             "
echo "============================================================================================"
mkdir checkpoints/No-6-2-53-translation-lm-dy
cp checkpoints/No-6-2-50-translation-dy/checkpoint_*_70000.pt checkpoints/No-6-2-53-translation-lm-dy/checkpoint_last.pt
bash call_scripts/iwslt14_de-en/training/No-6-2-53-translation-lm-dy.sh
