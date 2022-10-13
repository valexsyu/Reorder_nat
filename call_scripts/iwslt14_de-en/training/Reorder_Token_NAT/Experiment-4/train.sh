echo "============================================================================================"
echo "                   training              No-0-1                                             "
echo "============================================================================================"
bash call_scripts/training/No-0-1-translation.sh
mkdir checkpoints/No-0-2-reorder 
echo "============================================================================================"
echo "                   training              No-0-2                                             "
echo "============================================================================================"
cp checkpoints/No-0-1-translation/checkpoint_last.pt checkpoints/No-0-2-reorder/
bash call_scripts/training/No-0-2-reorder.sh
mkdir checkpoints/No-0-3-reorder_translation 
echo "============================================================================================"
echo "                   training              No-0-3                                             "
echo "============================================================================================"
cp checkpoints/No-0-2-reorder/checkpoint_last.pt checkpoints/No-0-3-reorder_translation/
bash call_scripts/training/No-0-3-reorder_translation.sh