DISTALL_DATA_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/mbert/mbert/demose
TOKEN_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/iwslt14_de_en_bibertDist_mbert_pruned26458/voc_8k_model
MODEL_NAME=/home/valexsyu/Doc/NMT/Reorder_nat/data/mbert/pruned_models_BertModel/pruned_V26458/vocab.txt
SRC=de
TGT=en
## tokenize translation data
mkdir $TOKEN_PATH

VOC_8k_MODEL=/home/valexsyu/Doc/NMT/Reorder_nat/data/mbert/pruned_models_BertModel/pruned_V26458/8k_vocab_models
cat $DISTALL_DATA_PATH/train.en $DISTALL_DATA_PATH/valid.en | shuf > $DISTALL_DATA_PATH/train.all
mkdir $VOC_8k_MODEL
python vocab_trainer.py --data $DISTALL_DATA_PATH/train.all --size 8000 --output ${VOC_8k_MODEL}


for prefix in "valid" "test" "train" ;
do
    for lang in $TGT ;
    do
        python transform_tokenize.py --input $DISTALL_DATA_PATH/${prefix}.${lang} --output $TOKEN_PATH/${prefix}.${lang} --pretrained_model $VOC_8k_MODEL
    done
done
for prefix in "valid" "test" "train" ;
do
    for lang in $SRC ;
    do
        python transform_tokenize.py --input $DISTALL_DATA_PATH/${prefix}.${lang} --output $TOKEN_PATH/${prefix}.${lang} --pretrained_model $MODEL_NAME
    done
done



# for prefix in "valid" "test" "train" ;
# do
#     for lang in $SRC $TGT ;
#     do
#         python transform_tokenize.py --input $DISTALL_DATA_PATH/${prefix}.${lang} --output $TOKEN_PATH/${prefix}.${lang} --pretrained_model $MODEL_NAME
#     done
# done


## get src and tgt vocabulary
cp /home/valexsyu/Doc/NMT/Reorder_nat/data/mbert/pruned_models_BertModel/pruned_V26458/vocab.txt $TOKEN_PATH/src_vocab.txt
cp $VOC_8k_MODEL/vocab.txt  $TOKEN_PATH/tgt_vocab.txt

