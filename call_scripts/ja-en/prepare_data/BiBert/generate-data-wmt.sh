cd /home/valexsyu/Doc/NMT/BiBERT
DATAPATH=./download_prepare/wmt-data/
STPATH=${DATAPATH}de-en-databin/
MODELPATH=./models/one-way-wmt/ 
PRE_SRC=jhu-clsp/bibert-ende
PRE=jhu-clsp/bibert-ende
TGT=en
SRC=de

for prefix in "valid" "train" ; do 
    CUDA_VISIBLE_DEVICES=0 fairseq-generate \
    ${STPATH} --path ${MODELPATH}checkpoint_best.pt --bpe bert --pretrained_bpe ${PRE} --pretrained_bpe_src ${PRE_SRC} \
    --beam 4 --lenpen 0.6 --remove-bpe --vocab_file=${STPATH}/dict.en.txt \
    --gen-subset $prefix |tee ${STPATH}/generate-$prefix.out
done



DISTILLPATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/bibert/wmt14_detoken_distill_awesome-align
mkdir -p $DISTILLPATH
for prefix in "valid" "train" ; do 
    sed -n '/^S-.*/ p' < ${STPATH}/generate-$prefix.out > $DISTILLPATH/$prefix.$SRC
    sed -n '/^D-.*/ p' < ${STPATH}/generate-$prefix.out > $DISTILLPATH/$prefix.$TGT
done

for prefix in "valid" "train" ; do 
    sed -i 's/^S-.*\t//g' $DISTILLPATH/$prefix.$SRC
    sed -i 's/^D-.*\t//g' $DISTILLPATH/$prefix.$TGT
done
TESTDATA=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/bibert/wmt14_detoken_awesome-align
cp $TESTDATA/test.* $DISTILLPATH
