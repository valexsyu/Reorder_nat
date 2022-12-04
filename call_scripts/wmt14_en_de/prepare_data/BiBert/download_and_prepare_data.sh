source $HOME/.bashrc 
conda activate bibert

# DISTALL_DATA_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_detoken_distill_wmt14_en_de
# TOKEN_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_token_distill_wmt14_en_de_52k
# MODEL_NAME=jhu-clsp/bibert-ende
# ## tokenize translation data
# mkdir $TOKEN_PATH
# cd /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/wmt14_en-de/prepare_data/BiBert
# for prefix in "valid" "test" "train" ;
# do
#     for lang in "en" "de" ;
#     do
#         python transform_tokenize.py --input $DISTALL_DATA_PATH/${prefix}.${lang} --output $TOKEN_PATH/${prefix}.${lang} --pretrained_model $MODEL_NAME
#     done
# done

# ## get src and tgt vocabulary
# python get_vocab.py --tokenizer jhu-clsp/bibert-ende --output ${TOKEN_PATH}/src_vocab.txt
# cp $TOKEN_PATH/src_vocab.txt $TOKEN_PATH/tgt_vocab.txt




# INPUT_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/baseline_detoken_distilled_wmt14_en_de
# TOKEN_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/baseline_token_distilled_wmt14_en_de_bibert
# MODEL_NAME=jhu-clsp/bibert-ende
# ## tokenize translation data
# mkdir $TOKEN_PATH
# cd /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/wmt14_en_de/prepare_data/BiBert


# SCRIPTS=/home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/tool/mosesdecoder/scripts

# ## de-mose data
# mkdir $INPUT_PATH/demose
# for prefix in "valid" "test" "train" ;
# do
#     for lang in "de" "en" ;
#     do
#         perl $SCRIPTS/tokenizer/detokenizer.perl -l en -q < $INPUT_PATH/${prefix}.${lang} > $INPUT_PATH/demose/${prefix}.${lang}
#     done
# done

# for prefix in "valid" "test" "train" ;
# do
#     for lang in "en" "de" ;
#     do
#         python transform_tokenize.py --input $INPUT_PATH/demose/${prefix}.${lang} --output $TOKEN_PATH/${prefix}.${lang} --pretrained_model $MODEL_NAME
#     done
# done

# ## get src and tgt vocabulary
# python get_vocab.py --tokenizer jhu-clsp/bibert-ende --output ${TOKEN_PATH}/src_vocab.txt
# cp $TOKEN_PATH/src_vocab.txt $TOKEN_PATH/tgt_vocab.txt



INPUT_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/BigBl_detoken_distill_wmt14_en_de
TOKEN_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/BigBl_token_distill_wmt14_en_de_bibert
MODEL_NAME=jhu-clsp/bibert-ende
## tokenize translation data
mkdir $TOKEN_PATH
cd /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/wmt14_en_de/prepare_data/BiBert


SCRIPTS=/home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/tool/mosesdecoder/scripts

## de-mose data
mkdir $INPUT_PATH/demose
for prefix in "train" ;
do
    for lang in "de" "en" ;
    do
        perl $SCRIPTS/tokenizer/detokenizer.perl -l en -q < $INPUT_PATH/${prefix}.${lang} > $INPUT_PATH/demose/${prefix}.${lang}
    done
done

for prefix in "valid" "test" "train" ;
do
    for lang in "en" "de" ;
    do
        python transform_tokenize.py --input $INPUT_PATH/demose/${prefix}.${lang} --output $TOKEN_PATH/${prefix}.${lang} --pretrained_model $MODEL_NAME
    done
done

## get src and tgt vocabulary
python get_vocab.py --tokenizer jhu-clsp/bibert-ende --output ${TOKEN_PATH}/src_vocab.txt
cp $TOKEN_PATH/src_vocab.txt $TOKEN_PATH/tgt_vocab.txt
