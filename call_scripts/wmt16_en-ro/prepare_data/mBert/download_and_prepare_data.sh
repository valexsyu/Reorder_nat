# INPUT_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/baseline_detoken_distilled_en-ro
# OUTPUT_TOKEN_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/baseline_token_distilled_wmt16_en_ro_mbert
# MODEL_NAME=bert-base-multilingual-uncased

# # echo 'Cloning Moses github repository (for tokenization scripts)...'
# # git clone https://github.com/moses-smt/mosesdecoder.git
# SCRIPTS=mosesdecoder/scripts


# ## de-mose data
# mkdir $INPUT_PATH/demose
# for prefix in "valid" "test" "train" ;
# do
#     for lang in "en" "ro" ;
#     do
#         perl $SCRIPTS/tokenizer/detokenizer.perl -l en -q < $INPUT_PATH/${prefix}.${lang} > $INPUT_PATH/demose/${prefix}.${lang}
#     done
# done

# ## tokenize translation data
# mkdir $OUTPUT_TOKEN_PATH
# for prefix in "valid" "test" "train" ;
# do
#     for lang in "en" "ro" ;
#     do
#         python transform_tokenize.py --input $INPUT_PATH/demose/${prefix}.${lang} --output $OUTPUT_TOKEN_PATH/${prefix}.${lang} --pretrained_model $MODEL_NAME
#     done
# done

# ## get src and tgt vocabulary 
# wget -O $OUTPUT_TOKEN_PATH/src_vocab.txt https://huggingface.co/bert-base-multilingual-uncased/resolve/main/vocab.txt
# cp $OUTPUT_TOKEN_PATH/src_vocab.txt $OUTPUT_TOKEN_PATH/tgt_vocab.txt

# INPUT_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/mbart_detoken_distilled_wmt16_en_ro
# OUTPUT_TOKEN_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/mbart_token_distilled_wmt16_en_ro_mbert
# MODEL_NAME=bert-base-multilingual-uncased

# # echo 'Cloning Moses github repository (for tokenization scripts)...'
# # git clone https://github.com/moses-smt/mosesdecoder.git
# SCRIPTS=mosesdecoder/scripts


# ## de-mose data
# mkdir $INPUT_PATH/demose
# for prefix in "valid" "test" "train" ;
# do
#     for lang in "en" "ro" ;
#     do
#         perl $SCRIPTS/tokenizer/detokenizer.perl -l en -q < $INPUT_PATH/${prefix}.${lang} > $INPUT_PATH/demose/${prefix}.${lang}
#     done
# done

# ## tokenize translation data
# mkdir $OUTPUT_TOKEN_PATH
# for prefix in "valid" "test" "train" ;
# do
#     for lang in "en" "ro" ;
#     do
#         python transform_tokenize.py --input $INPUT_PATH/demose/${prefix}.${lang} --output $OUTPUT_TOKEN_PATH/${prefix}.${lang} --pretrained_model $MODEL_NAME
#     done
# done

# ## get src and tgt vocabulary 
# wget -O $OUTPUT_TOKEN_PATH/src_vocab.txt https://huggingface.co/bert-base-multilingual-uncased/resolve/main/vocab.txt
# cp $OUTPUT_TOKEN_PATH/src_vocab.txt $OUTPUT_TOKEN_PATH/tgt_vocab.txt



# INPUT_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/wmt16_en_ro_detoken
# OUTPUT_TOKEN_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/wmt16_en_ro_token_mbert
# MODEL_NAME=bert-base-multilingual-uncased

# # echo 'Cloning Moses github repository (for tokenization scripts)...'
# # git clone https://github.com/moses-smt/mosesdecoder.gits
# SCRIPTS=mosesdecoder/scripts


# ## de-mose data
# mkdir $INPUT_PATH/demose
# for prefix in "valid" "test" "train" ;
# do
#     for lang in "en" "ro" ;
#     do
#         perl $SCRIPTS/tokenizer/detokenizer.perl -l en -q < $INPUT_PATH/${prefix}.${lang} > $INPUT_PATH/demose/${prefix}.${lang}
#     done
# done

# ## tokenize translation data
# mkdir $OUTPUT_TOKEN_PATH
# for prefix in "valid" "test" "train" ;
# do
#     for lang in "en" "ro" ;
#     do
#         python transform_tokenize.py --input $INPUT_PATH/demose/${prefix}.${lang} --output $OUTPUT_TOKEN_PATH/${prefix}.${lang} --pretrained_model $MODEL_NAME
#     done
# done

# ## get src and tgt vocabulary 
# wget -O $OUTPUT_TOKEN_PATH/src_vocab.txt https://huggingface.co/bert-base-multilingual-uncased/resolve/main/vocab.txt
# cp $OUTPUT_TOKEN_PATH/src_vocab.txt $OUTPUT_TOKEN_PATH/tgt_vocab.txt












INPUT_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/wmt16_en_ro_BigBlDist_detoken
OUTPUT_TOKEN_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/wmt16_en_ro_BigBlDist_mbert
MODEL_NAME=bert-base-multilingual-uncased

# echo 'Cloning Moses github repository (for tokenization scripts)...'
# git clone https://github.com/moses-smt/mosesdecoder.gits
# SCRIPTS=mosesdecoder/scripts


# ## de-mose data
# mkdir $INPUT_PATH/demose
# for prefix in "train" ;
# do
#     for lang in "en" "ro" ;
#     do
#         echo "demose ${prefix}.${lang} "
#         perl $SCRIPTS/tokenizer/detokenizer.perl -l en -q < $INPUT_PATH/${prefix}.${lang} > $INPUT_PATH/demose/${prefix}.${lang}
#     done
# done

## tokenize translation data
mkdir $OUTPUT_TOKEN_PATH
for prefix in "train" ;
do
    for lang in "en" "ro" ;
    do 
        echo "tokenize ${prefix}.${lang} "
        python transform_tokenize.py --input $INPUT_PATH/demose/${prefix}.${lang} --output $OUTPUT_TOKEN_PATH/${prefix}.${lang} --pretrained_model $MODEL_NAME
    done
done

## get src and tgt vocabulary 
wget -O $OUTPUT_TOKEN_PATH/src_vocab.txt https://huggingface.co/bert-base-multilingual-uncased/resolve/main/vocab.txt
cp $OUTPUT_TOKEN_PATH/src_vocab.txt $OUTPUT_TOKEN_PATH/tgt_vocab.txt

