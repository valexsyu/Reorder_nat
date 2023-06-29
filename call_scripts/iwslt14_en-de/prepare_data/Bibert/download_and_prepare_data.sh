source $HOME/.bashrc 
conda activate bibert

# DISTALL_DATA_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_detoken_distill_iwslt14_en_de
# TOKEN_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_token_distill_iwslt14_en_de_bibert
# MODEL_NAME=jhu-clsp/bibert-ende
# ## tokenize translation data
# mkdir $TOKEN_PATH
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


# DISTALL_DATA_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/baseline_detoken_distill_iwslt14_en_de
# TOKEN_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/baseline_token_distill_iwslt14_en_de_bibert
# MODEL_NAME=jhu-clsp/bibert-ende

# cd /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/iwslt14_en-de/prepare_data/Bibert

# SCRIPTS=/home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/tool/mosesdecoder/scripts

# ## de-mose data
# mkdir $INPUT_PATH/demose
# for prefix in "train" ;
# do
#     for lang in "de" "en" ;
#     do
#         perl $SCRIPTS/tokenizer/detokenizer.perl -l en -q < $INPUT_PATH/${prefix}.${lang} > $INPUT_PATH/demose/${prefix}.${lang}
#     done
# done


# ## tokenize translation data
# mkdir $TOKEN_PATH
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


#===================Use the Pruned model 2023/06/27 Bibert with the Bibert distilled dataset Note: use the corrected test dataset =================
DISTALL_DATA_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_detoken_distill_iwslt14_en_de
TOKEN_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/iwslt14_en_de_bibertDist_bibert
PRUN_MODEL_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/iwslt14_en_de_bibertDist_bibert/src_vocab.txt
MODEL_NAME=jhu-clsp/bibert-ende
mkdir $TOKEN_PATH

# for prefix in "valid" "test" "train" ;
# do
#     for lang in "en" "de" ;
#     do
#         echo "${prefix}.${lang} is processing"
#         python transform_tokenize.py --input $DISTALL_DATA_PATH/${prefix}.${lang} --output $TOKEN_PATH/${prefix}.${lang} --pretrained_model $PRUN_MODEL_PATH
#     done
# done
for prefix in "test";
do
    for lang in "en" "de" ;
    do
        echo "${prefix}.${lang} is processing"
        python transform_tokenize.py --input $DISTALL_DATA_PATH/${prefix}.${lang} --output $TOKEN_PATH/${prefix}.${lang} --pretrained_model $PRUN_MODEL_PATH
    done
done


# cp $PRUN_MODEL_PATH/vocab.txt $TOKEN_PATH/tgt_vocab.txt
# cp $PRUN_MODEL_PATH/vocab.txt $TOKEN_PATH/src_vocab.txt

source $HOME/.bashrc 
conda activate bibert

TEXT=$TOKEN_PATH

fairseq-preprocess --source-lang en --target-lang de  --trainpref $TEXT/train --validpref $TEXT/valid \
--testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
--tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25 --align-suffix align \