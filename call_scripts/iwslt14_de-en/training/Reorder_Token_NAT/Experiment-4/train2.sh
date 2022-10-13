# echo "============================================================================================"
# echo "                   training              No-2-1                                             "
# echo "============================================================================================"
# bash call_scripts/training/No-2-1-translation.sh
# mkdir checkpoints/No-2-2-reorder
# echo "============================================================================================"
# echo "                   training              No-2-2                                             "
# echo "============================================================================================"
# cp checkpoints/No-2-1-translation/checkpoint_last.pt checkpoints/No-2-2-reorder/
# bash call_scripts/training/No-2-2-reorder.sh
# mkdir checkpoints/No-2-3-reorder_translation 
echo "============================================================================================"
echo "                   training              No-2-3                                             "
echo "============================================================================================"
# cp checkpoints/No-2-2-reorder/checkpoint_last.pt checkpoints/No-2-3-reorder_translation/
bash call_scripts/training/No-2-3-reorder_translation.sh
echo "============================================================================================"
echo "                   training              No-2-1-0                                           "
echo "============================================================================================"
bash call_scripts/training/No-2-1-0-translation-wo_align.sh
# echo "============================================================================================"
# echo "                   training              No-2-1-2                                             "
# echo "============================================================================================"
# bash call_scripts/training/No-2-1-2-translation_small-wo_align.sh