DISTALL_DATA_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_detoken_distill_wmt14_de_en
TOKEN_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_token_distill_wmt14_de_en_dual-bert-de-en
MODEL_NAME_EN=bert-base-uncased
MODEL_NAME_DE=dbmdz/bert-base-german-uncased
## tokenize translatio
mkdir $TOKEN_PATH
cd /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/wmt14_de-en/prepare_data/Bert
# for prefix in "valid" "test" "train" ;
# do
#     for lang in "en" ;
#     do
#         python transform_tokenize.py --input $DISTALL_DATA_PATH/${prefix}.${lang} --output $TOKEN_PATH/${prefix}.${lang} --pretrained_model $MODEL_NAME_EN
#     done
# done

for prefix in "valid" "test" "train" ;
do
    for lang in "de" ;
    do
        python transform_tokenize.py --input $DISTALL_DATA_PATH/${prefix}.${lang} --output $TOKEN_PATH/${prefix}.${lang} --pretrained_model $MODEL_NAME_DE
    done
done

## get src and tgt vocabulary
wget -O $TOKEN_PATH/tgt_vocab.txt https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt
wget -O $TOKEN_PATH/src_vocab.txt https://huggingface.co/dbmdz/bert-base-german-uncased/resolve/main/vocab.txt
