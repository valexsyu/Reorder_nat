source $HOME/.bashrc 
conda activate base
echo "============================================================================================"
echo "                   training              No-6-5-00                                             "
echo "============================================================================================"
#bash call_scripts/iwslt14_de-en/training/No-6-5-00-translation.sh
echo "============================================================================================"
echo "                   training              No-6-5-03                                             "
echo "============================================================================================"
mkdir checkpoints/No-6-5-03-translation-self_lm 
cp checkpoints/No-6-5-00-translation/checkpoint_*_70000.pt checkpoints/No-6-5-03-translation-self_lm/checkpoint_last.pt
bash call_scripts/iwslt14_de-en/training/No-6-5-03-translation-self_lm.sh
