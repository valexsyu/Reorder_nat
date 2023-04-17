#!/bin/bash
source $HOME/.bashrc 
conda activate base

DISTALL_DATA_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_detoken_distill_iwslt14_de_en
TOKEN_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/iwslt14_de_en_bibertDist_xlmr_pruned21785
PRUN_MODEL_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/pruned_model/xlmr/pruned_models/pruned_V21785
## tokenize translation data
mkdir $TOKEN_PATH

for prefix in "valid" "test" "train" ;
do
    for lang in "en" "de" ;
    do
        echo "${prefix}.${lang} is processing"
        python transform_tokenize.py --input $DISTALL_DATA_PATH/${prefix}.${lang} --output $TOKEN_PATH/${prefix}.${lang} --pretrained_model $PRUN_MODEL_PATH
    done
done



cp $PRUN_MODEL_PATH/vocab.txt $TOKEN_PATH/tgt_vocab.txt
cp $PRUN_MODEL_PATH/vocab.txt $TOKEN_PATH/src_vocab.txt


TEXT=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/iwslt14_de_en_bibertDist_mbert_pruned26458

fairseq-preprocess --source-lang de --target-lang en  --trainpref $TEXT/train --validpref $TEXT/valid \
--testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
--tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25 --align-suffix align \