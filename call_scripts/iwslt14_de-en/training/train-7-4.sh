source $HOME/.bashrc 
conda activate base
# echo "============================================================================================"
# echo "                   training              No-7-4-04                                           "
# echo "============================================================================================"
# bash call_scripts/iwslt14_de-en/training/No-7-4-04-translation-valid.sh
# echo "============================================================================================"
# echo "                   training              No-7-4-05                                             "
# echo "============================================================================================"
# mkdir checkpoints/No-7-4-05-translation-lm-valid
# cp checkpoints/No-7-4-04-translation-valid/checkpoint_*_70000.pt checkpoints/No-7-4-05-translation-lm-valid/checkpoint_last.pt
# bash call_scripts/iwslt14_de-en/training/No-7-4-05-translation-lm-valid.sh
echo "============================================================================================"
echo "                   training              No-7-4-00                                           "
echo "============================================================================================"
bash call_scripts/iwslt14_de-en/training/No-7-4-00-translation.sh
echo "============================================================================================"
echo "                   training              No-7-4-03                                             "
echo "============================================================================================"
mkdir checkpoints/No-7-4-03-translation-lm
cp checkpoints/No-7-4-00-translation/checkpoint_*_70000.pt checkpoints/No-7-4-03-translation-lm/checkpoint_last.pt
bash call_scripts/iwslt14_de-en/training/No-7-4-03-translation-lm.sh
# echo "============================================================================================"
# echo "                   training              No-7-4-04-MSE                                           "
# echo "============================================================================================"
# bash call_scripts/iwslt14_de-en/training/No-7-4-04-translation-lm_MSE.sh