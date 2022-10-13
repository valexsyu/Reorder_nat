# echo "============================================================================================"
# echo "                   training              No-3-1                                             "
# echo "============================================================================================"
# bash call_scripts/iwslt14_de-en/training/No-3-1-1-translation.sh
# echo "============================================================================================"
# echo "                   training              No-3-2                                             "
# echo "============================================================================================"
# mkdir checkpoints/No-3-2-1-reorder
# cp checkpoints/No-3-1-1-translation/checkpoint_last.pt checkpoints/No-3-2-1-reorder/
# bash call_scripts/iwslt14_de-en/training/No-3-2-1-reorder.sh
# echo "============================================================================================"
echo "                   training              No-3-3                                             "
echo "============================================================================================"
# mkdir checkpoints/No-3-3-1-reorder_translation 
# cp checkpoints/No-3-2-1-reorder/checkpoint_last.pt checkpoints/No-3-3-1-reorder_translation/
bash call_scripts/iwslt14_de-en/training/No-3-3-1-reorder_translation.sh
echo "============================================================================================"
echo "                   training              No-3-4                                             "
echo "============================================================================================"
mkdir checkpoints/No-3-4-1-reorder_translation 
cp checkpoints/No-3-3-1-reorder/checkpoint_last.pt checkpoints/No-3-4-1-reorder_translation/
bash call_scripts/iwslt14_de-en/training/No-3-4-1-reorder_translation.sh
# echo "============================================================================================"
# echo "                   training              No-3-1-0                                           "
# echo "============================================================================================"
# bash call_scripts/training/No-3-1-0-translation-wo_align.sh
# echo "============================================================================================"
# echo "                   training              No-3-1-2                                             "
# echo "============================================================================================"
# bash call_scripts/training/No-3-1-2-translation_small-wo_align.sh