echo "============================================================================================"
echo "                   training              No-4-1-1                                           "
echo "============================================================================================"
bash call_scripts/iwslt14_de-en/training/No-4-1-1-translation_emb.sh
echo "============================================================================================"
echo "                   training              No-4-2-1                                           "
echo "============================================================================================"
mkdir checkpoints/No-4-2-1-reorder_emb
cp checkpoints/No-4-1-1-translation_emb/checkpoint_last.pt checkpoints/No-4-2-1-reorder_emb/
bash call_scripts/iwslt14_de-en/training/No-4-2-1-reorder_emb.sh
echo "============================================================================================"
echo "                   training              No-4-4-1                                             "
echo "============================================================================================"
mkdir checkpoints/No-4-4-1-reorder_translation_lm_emb
cp checkpoints/No-4-2-1-reorder_emb/checkpoint_last.pt checkpoints/No-4-4-1-reorder_translation_lm_emb/
bash call_scripts/iwslt14_de-en/training/No-4-4-1-reorder_translation_lm_emb.sh
# echo "============================================================================================"
# echo "                   training              No-3-4                                             "
# echo "============================================================================================"
# mkdir checkpoints/No-3-4-1-reorder_translation 
# cp checkpoints/No-3-3-1-reorder/checkpoint_last.pt checkpoints/No-3-4-1-reorder_translation/
# bash call_scripts/iwslt14_de-en/training/No-3-4-1-reorder_translation.sh
# echo "============================================================================================"
# echo "                   training              No-3-1-0                                           "
# echo "============================================================================================"
# bash call_scripts/training/No-3-1-0-translation-wo_align.sh
# echo "============================================================================================"
# echo "                   training              No-3-1-2                                             "
# echo "============================================================================================"
# bash call_scripts/training/No-3-1-2-translation_small-wo_align.sh