echo "============================================================================================"
echo "                   training              No-7-1-2                                           "
echo "============================================================================================"
bash call_scripts/iwslt14_de-en/training/No-7-1-2-translation_emb.sh
echo "============================================================================================"
echo "                   training              No-7-2-2                                           "
echo "============================================================================================"
mkdir checkpoints/No-7-2-2-reorder_emb
cp checkpoints/No-7-1-2-translation_emb/checkpoint_last.pt checkpoints/No-7-2-2-reorder_emb/
bash call_scripts/iwslt14_de-en/training/No-7-2-2-reorder_emb.sh
echo "============================================================================================"
echo "                   training              No-7-4-2                                             "
echo "============================================================================================"
mkdir checkpoints/No-7-4-2-reorder_translation_lm-dis_emb
cp checkpoints/No-7-2-2-reorder_emb/checkpoint_last.pt checkpoints/No-7-4-2-reorder_translation_lm-dis_emb/
bash call_scripts/iwslt14_de-en/training/No-7-4-2-reorder_translation_lm-dis_emb.sh