DISTALL_DATA_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_detoken_distill_iwslt14_de_en
TOKEN_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_token_distill_iwslt14_de_en_distill-mbert-1
MODEL_NAME=distilbert-base-multilingual-cased
## tokenize translation data
mkdir $TOKEN_PATH
cd /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/iwslt14_de-en/prepare_data/distill-mBert
for prefix in "valid" "test" "train" ;
do
    for lang in "en" "de" ;
    do
        python transform_tokenize.py --input $DISTALL_DATA_PATH/${prefix}.${lang} --output $TOKEN_PATH/${prefix}.${lang} --pretrained_model $MODEL_NAME
    done
done

## get src and tgt vocabulary
wget -O $TOKEN_PATH/src_vocab.txt https://huggingface.co/distilbert-base-multilingual-cased/resolve/main/vocab.txt
