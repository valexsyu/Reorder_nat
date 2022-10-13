databin_dir=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/bibert/wmt20_jaen_filter09995/ja-en.distill
OUTPATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/bibert/wmt20_jaen
mkdir -p $OUTPATH

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    $databin_dir \
    --save-dir $OUTPATH \
    --arch transformer_iwslt_de_en  --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --max-update 200 \
    --save-interval-updates 50 \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric