databin_dir=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/bibert/wmt20_jaen_filter09995/ja-en.distill
OUTPATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/bibert/wmt20_jaen

CUDA_VISIBLE_DEVICES=0 fairseq-generate \
    $databin_dir \
    --path $OUTPATH/checkpoint_last.pt \
    --gen-subset valid \
    --beam 1 --remove-bpe --batch-size 500 |tee ${OUTPATH}/generate-valid.out