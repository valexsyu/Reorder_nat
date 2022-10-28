DISTALL_DATA_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_detoken_distill_valid-nondistill_iwslt14_de_en
TOKEN_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_token_distill_valid-nondistill_iwslt14_de_en_xlmr
MODEL_NAME=xlm-roberta-base
## tokenize translation data
mkdir $TOKEN_PATH
cd /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/iwslt14_de-en/prepare_data/XLMR
for prefix in "valid" "test" "train" ;
do
    for lang in "en" "de" ;
    do
        python transform_tokenize.py --input $DISTALL_DATA_PATH/${prefix}.${lang} --output $TOKEN_PATH/${prefix}.${lang} --pretrained_model $MODEL_NAME
    done
done

python get_vocab.py --tokenizer $MODEL_NAME --output $TOKEN_PATH/src_vocab.txt
