# DISTALL_DATA_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_detoken_distill_wmt14_de_en
# TOKEN_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_token_distill_wmt14_de_en_52k
# MODEL_NAME=jhu-clsp/bibert-ende
# ## tokenize translation data
# mkdir $TOKEN_PATH
# cd /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/wmt14_de_en/prepare_data/BiBert
# for prefix in "train" ; # "valid" "test" "train" ;
# do
#     for lang in "en" ; # "en" "de" ;
#     do
#         python transform_tokenize.py --input $DISTALL_DATA_PATH/${prefix}.${lang} --output $TOKEN_PATH/${prefix}.${lang} --pretrained_model $MODEL_NAME
#     done
# done

# ## get src and tgt vocabulary
# python get_vocab.py --tokenizer jhu-clsp/bibert-ende --output ${TOKEN_PATH}/src_vocab.txt
# cp $TOKEN_PATH/src_vocab.txt $TOKEN_PATH/tgt_vocab.txt



DISTALL_DATA_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/wmt14_de-en_detoken_nondistill
TOKEN_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/wmt14_de_en_token_bibert
MODEL_NAME=jhu-clsp/bibert-ende
## tokenize translation data
mkdir $TOKEN_PATH
cd /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/wmt14_de_en/prepare_data/BiBert
for prefix in "train" ; # "valid" "test" "train" ;
do
    for lang in "de" ; # "en" "de" ;
    do
        python transform_tokenize.py --input $DISTALL_DATA_PATH/${prefix}.${lang} --output $TOKEN_PATH/${prefix}.${lang} --pretrained_model $MODEL_NAME
    done
done

for prefix in "valid" "test" ; # "valid" "test" "train" ;
do
    for lang in "en" "de" ;
    do
        python transform_tokenize.py --input $DISTALL_DATA_PATH/${prefix}.${lang} --output $TOKEN_PATH/${prefix}.${lang} --pretrained_model $MODEL_NAME
    done
done

## get src and tgt vocabulary
python get_vocab.py --tokenizer jhu-clsp/bibert-ende --output ${TOKEN_PATH}/src_vocab.txt
cp $TOKEN_PATH/src_vocab.txt $TOKEN_PATH/tgt_vocab.txt
cp $TOKEN_PATH/* /home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/wmt14_en_de_token_bibert/