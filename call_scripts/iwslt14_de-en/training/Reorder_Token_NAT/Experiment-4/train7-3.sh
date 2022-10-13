echo "============================================================================================"
echo "                   training              No-7-5-1                                             "
echo "============================================================================================"
mkdir checkpoints/No-7-5-1-reorder_translation_emb
cp checkpoints/No-7-2-1-reorder_emb/checkpoint_last.pt checkpoints/No-7-5-1-reorder_translation_emb/
bash call_scripts/iwslt14_de-en/training/No-7-5-1-reorder_translation_emb.sh