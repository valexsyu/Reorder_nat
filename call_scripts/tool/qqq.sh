if [ -e checkpoints/$1/checkpoint_last.pt ] && \
   [ $(ls checkpoints/$1/checkpoint.best_bleu_* 2>/dev/null | grep -c "^checkpoints/$1/checkpoint.best_bleu_.*") -eq 5 ]; then
    echo "All 6 checkpoint files exist"
else
    echo "Not all 6 checkpoint files exist"
fi
