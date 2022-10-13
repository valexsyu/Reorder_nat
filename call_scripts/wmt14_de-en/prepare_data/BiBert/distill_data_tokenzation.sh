DISTILL_DATA_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/bibert/wmt14_detoken_distill_awesome-align
VOCAB_MODELS_8K=$DISTILL_DATA_PATH/8k-vocab-models
VOCAB_MODELS_12K=$DISTILL_DATA_PATH/12k-vocab-models
BIBERT_TOK=$DISTILL_DATA_PATH/bibert_tok
TOK_8K=$DISTILL_DATA_PATH/8k_tok
TOK_12K=$DISTILL_DATA_PATH/12k_tok
DATA=$DISTILL_DATA_PATH/data
DATA_MIXED_FT=$DISTILL_DATA_PATH/data_mixed_ft
DATA_MIXED=$DISTILL_DATA_PATH/data_mixed
## train 8K tokenizer for ordinary translation:
cat $DISTILL_DATA_PATH/train.en $DISTILL_DATA_PATH/valid.en $DISTILL_DATA_PATH/test.en | shuf > $DISTILL_DATA_PATH/train.all

cd /home/valexsyu/Doc/NMT/BiBERT/download_prepare
mkdir ${VOCAB_MODELS_8K}

python vocab_trainer.py --data $DISTILL_DATA_PATH/train.all --size 8000 --output ${VOCAB_MODELS_8K}

## train 12K tokenizer for dual-directional translation
cat $DISTILL_DATA_PATH/train.en $DISTILL_DATA_PATH/valid.en $DISTILL_DATA_PATH/test.en $DISTILL_DATA_PATH/train.de $DISTILL_DATA_PATH/valid.de $DISTILL_DATA_PATH/test.de | shuf > $DISTILL_DATA_PATH/train.all.dual
mkdir ${VOCAB_MODELS_12K}
python vocab_trainer.py --data $DISTILL_DATA_PATH/train.all.dual --size 12000 --output ${VOCAB_MODELS_12K}



## tokenize translation data
mkdir $BIBERT_TOK
mkdir ${TOK_8K}
mkdir ${TOK_12K}

for prefix in "valid" "test" "train" ;
do
    for lang in "en" "de" ;
    do
        python transform_tokenize.py --input $DISTILL_DATA_PATH/${prefix}.${lang} --output $BIBERT_TOK/${prefix}.${lang} --pretrained_model jhu-clsp/bibert-ende
    done
done

for prefix in "valid" "test" "train" ;
do
    python transform_tokenize.py --input $DISTILL_DATA_PATH/${prefix}.en --output ${TOK_8K}/${prefix}.en --pretrained_model ${VOCAB_MODELS_8K}
done


for prefix in "valid" "test" "train" ;
do
    for lang in "en" "de";
    do
    python transform_tokenize.py --input $DISTILL_DATA_PATH/${prefix}.${lang} --output ${TOK_12K}/${prefix}.${lang} --pretrained_model ${VOCAB_MODELS_12K}
    done
done


mkdir ${DATA}   # for one-way translation data
cp $BIBERT_TOK/*.de ${DATA}/ 
cp ${TOK_8K}/*.en ${DATA}/



mkdir $DATA_MIXED_FT # for dual-directional fine-tuning data. we first preprocess this because it will be easier to finish
cp $BIBERT_TOK/*.de $DATA_MIXED_FT/
cp ${TOK_12K}/*.en $DATA_MIXED_FT/

mkdir $DATA_MIXED # preprocess dual-directional data

cd $DATA_MIXED
cat $BIBERT_TOK/train.en $BIBERT_TOK/train.de > train.all.en
cat ${TOK_12K}/train.de ${TOK_12K}/train.en > train.all.de
paste -d '@@@' train.all.en /dev/null /dev/null train.all.de | shuf > train.all
cat train.all | awk -F'@@@' '{print $1}' > train.de
cat train.all | awk -F'@@@' '{print $2}' > train.en
rm train.all*

cat $BIBERT_TOK/valid.en $BIBERT_TOK/valid.de > valid.all.en
cat ${TOK_12K}/valid.de ${TOK_12K}/valid.en > valid.all.de
paste -d '@@@' valid.all.en /dev/null /dev/null valid.all.de | shuf > valid.all
cat valid.all | awk -F'@@@' '{print $1}' > valid.de
cat valid.all | awk -F'@@@' '{print $2}' > valid.en
rm valid.all*

cp $BIBERT_TOK/test.de .
cp ${TOK_12K}/test.en .
cd /home/valexsyu/Doc/NMT/BiBERT/download_prepare




## get src and tgt vocabulary
python get_vocab.py --tokenizer jhu-clsp/bibert-ende --output ${DATA}/src_vocab.txt
python get_vocab.py --tokenizer jhu-clsp/bibert-ende --output $DATA_MIXED/src_vocab.txt
python get_vocab.py --tokenizer jhu-clsp/bibert-ende --output $DATA_MIXED_FT/src_vocab.txt
python get_vocab.py --tokenizer jhu-clsp/bibert-ende --output $BIBERT_TOK/src_vocab.txt
python get_vocab.py --tokenizer ${VOCAB_MODELS_8K} --output ${DATA}/tgt_vocab.txt
python get_vocab.py --tokenizer ${VOCAB_MODELS_12K} --output $DATA_MIXED/tgt_vocab.txt
python get_vocab.py --tokenizer ${VOCAB_MODELS_12K} --output $DATA_MIXED_FT/tgt_vocab.txt
python get_vocab.py --tokenizer jhu-clsp/bibert-ende --output $BIBERT_TOK/tgt_vocab.txt


## remove useless files
# rm -rf $BIBERT_TOK
rm -rf ${TOK_8K}
rm -rf ${TOK_12K}









