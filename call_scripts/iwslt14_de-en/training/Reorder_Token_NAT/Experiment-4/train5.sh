echo "============================================================================================"
echo "                   training              No-5-1-1                                           "
echo "============================================================================================"
bash call_scripts/iwslt14_de-en/training/No-5-1-1-translation_emb.sh
echo "============================================================================================"
echo "                   training              No-5-2-1                                           "
echo "============================================================================================"
mkdir checkpoints/No-5-2-1-reorder_emb
cp checkpoints/No-5-1-1-translation_emb/checkpoint_last.pt checkpoints/No-5-2-1-reorder_emb/
bash call_scripts/iwslt14_de-en/training/No-5-2-1-reorder_emb.sh
echo "============================================================================================"
echo "                   training              No-5-4-1                                             "
echo "============================================================================================"
mkdir checkpoints/No-5-4-1-reorder_translation_lm_emb
cp checkpoints/No-5-2-1-reorder_emb/checkpoint_last.pt checkpoints/No-5-4-1-reorder_translation_lm_emb/
bash call_scripts/iwslt14_de-en/training/No-5-4-1-reorder_translation_lm_emb.sh
