# echo "============================================================================================"
# echo "                   training              No-2-1                                             "
# echo "============================================================================================"
# bash call_scripts/iwslt14_de-en/training/No-2-1-translation_1GPU.sh 
# echo "============================================================================================"
# echo "                   training              No-2-2                                             "
# echo "============================================================================================"
# mkdir checkpoints/No-2-2-reorder_1GPU
# cp checkpoints/No-2-1-translation_1GPU/checkpoint_last.pt checkpoints/No-2-2-reorder_1GPU/
# bash call_scripts/iwslt14_de-en/training/No-2-2-reorder_1GPU.sh
# echo "============================================================================================"
# echo "                   training              No-2-3                                             "
# echo "============================================================================================"
# mkdir checkpoints/No-2-3-reorder_translation_1GPU 
# cp checkpoints/No-2-2-reorder_1GPU/checkpoint_last.pt checkpoints/No-2-3-reorder_translation_1GPU/
# bash call_scripts/iwslt14_de-en/training/No-2-3-reorder_translation_1GPU.sh
echo "============================================================================================"
echo "                   training              No-2-4                                             "
echo "============================================================================================"
mkdir checkpoints/No-2-4-reorder_translation_lm_1GPU 
cp checkpoints/No-2-3-reorder_translation_1GPU/checkpoint_last.pt checkpoints/No-2-4-reorder_translation_lm_1GPU/
bash call_scripts/iwslt14_de-en/training/No-2-4-reorder_translation_lm_1GPU.sh
# echo "============================================================================================"
# echo "                   training              No-2-1-0                                           "
# echo "============================================================================================"
# bash call_scripts/training/No-2-1-0-translation-wo_align.sh
# echo "============================================================================================"
# echo "                   training              No-2-1-2                                             "
# echo "============================================================================================"
# bash call_scripts/training/No-2-1-2-translation_small-wo_align.sh