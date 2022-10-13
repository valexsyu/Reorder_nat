# DISTALL_DATA_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_detoken_iwslt14_de_en
# TOKEN_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_token_iwslt14_de_en_52k
# MODEL_NAME_EN=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_token_iwslt14_de_en_52k/tgt_vocab.txt
# MODEL_NAME_DE=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_token_iwslt14_de_en_52k/src_vocab.txt
# ## tokenize translatio
# mkdir $TOKEN_PATH
# cd /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/iwslt14_de-en/prepare_data/BiBert
# for prefix in "valid" "test" "train" ;
# do
#     for lang in "en" ;
#     do
#         python transform_tokenize.py --input $DISTALL_DATA_PATH/${prefix}.${lang} --output $TOKEN_PATH/${prefix}.${lang} --pretrained_model $MODEL_NAME_EN
#     done
# done

# for prefix in "valid" "test" "train" ;
# do
#     for lang in "de" ;
#     do
#         python transform_tokenize.py --input $DISTALL_DATA_PATH/${prefix}.${lang} --output $TOKEN_PATH/${prefix}.${lang} --pretrained_model $MODEL_NAME_DE
#     done
# done

DISTALL_DATA_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_detoken_iwslt14_de_en
TOKEN_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_token_iwslt14_de_en_mbert
MODEL_NAME_EN=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_token_iwslt14_de_en_mbert/tgt_vocab.txt
MODEL_NAME_DE=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_token_iwslt14_de_en_mbert/src_vocab.txt
## tokenize translatio
mkdir $TOKEN_PATH
cd /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/iwslt14_de-en/prepare_data/BiBert
for prefix in "valid" "test" "train" ;
do
    for lang in "en" ;
    do
        python transform_tokenize.py --input $DISTALL_DATA_PATH/${prefix}.${lang} --output $TOKEN_PATH/${prefix}.${lang} --pretrained_model $MODEL_NAME_EN
    done
done

for prefix in "valid" "test" "train" ;
do
    for lang in "de" ;
    do
        python transform_tokenize.py --input $DISTALL_DATA_PATH/${prefix}.${lang} --output $TOKEN_PATH/${prefix}.${lang} --pretrained_model $MODEL_NAME_DE
    done
done