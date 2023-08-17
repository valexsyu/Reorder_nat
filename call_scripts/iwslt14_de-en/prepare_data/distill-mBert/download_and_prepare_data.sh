# DISTALL_DATA_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_detoken_distill_valid-nondistill_iwslt14_de_en
# TOKEN_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_token_distill_valid-nondistill_iwslt14_de_en_distill-mbert
# MODEL_NAME=distilbert-base-multilingual-cased
# ## tokenize translation data
# mkdir $TOKEN_PATH
# cd /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/iwslt14_de-en/prepare_data/distill-mBert
# for prefix in "valid" "test" "train" ;
# do
#     for lang in "en" "de" ;
#     do
#         python transform_tokenize.py --input $DISTALL_DATA_PATH/${prefix}.${lang} --output $TOKEN_PATH/${prefix}.${lang} --pretrained_model $MODEL_NAME
#     done
# done

# ## get src and tgt vocabulary
# wget -O $TOKEN_PATH/src_vocab.txt https://huggingface.co/distilbert-base-multilingual-cased/resolve/main/vocab.txt



# DISTALL_DATA_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/baseline_detoken_distill_valid-nondistill_iwslt14_de_en
# TOKEN_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/baseline_token_distill_valid-nondistill_iwslt14_de_en_distill-mbert
# MODEL_NAME=distilbert-base-multilingual-cased
# ## tokenize translation data
# mkdir $TOKEN_PATH
# cd /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/iwslt14_de-en/prepare_data/distill-mBert
# for prefix in "valid" "test" "train" ;
# do
#     for lang in "en" "de" ;
#     do
#         python transform_tokenize.py --input $DISTALL_DATA_PATH/${prefix}.${lang} --output $TOKEN_PATH/${prefix}.${lang} --pretrained_model $MODEL_NAME
#     done
# done

# ## get src and tgt vocabulary
# wget -O $TOKEN_PATH/src_vocab.txt https://huggingface.co/distilbert-base-multilingual-cased/resolve/main/vocab.txt
# cp $TOKEN_PATH/src_vocab.txt $TOKEN_PATH/tgt_vocab.txt


#===================Use the Pruned model 2023/08/14 regenerate the dataset to get correct test set Note: use the corrected test dataset =================
root=/home/valex/Documents/Study/battleship
DISTALL_DATA_PATH=$root/Reorder_nat/data/nat_position_reorder/awesome/Bibert_detoken_distill_iwslt14_de_en
TOKEN_PATH=$root/Reorder_nat/data/nat_position_reorder/awesome/iwslt14_de_en_bibertDist_dmbert
MODEL_NAME=distilbert-base-multilingual-cased
mkdir $TOKEN_PATH

for prefix in "valid" "test" "train" ;
do
    for lang in "en" "de" ;
    do
        echo "${prefix}.${lang} is processing"
        python transform_tokenize.py --input $DISTALL_DATA_PATH/${prefix}.${lang} --output $TOKEN_PATH/${prefix}.${lang} --pretrained_model $MODEL_NAME
    done
done


wget -O $TOKEN_PATH/src_vocab.txt https://huggingface.co/distilbert-base-multilingual-cased/resolve/main/vocab.txt
cp $TOKEN_PATH/src_vocab.txt $TOKEN_PATH/tgt_vocab.txt

# source $HOME/.bashrc 
# conda activate bibert

TEXT=$TOKEN_PATH

fairseq-preprocess --source-lang de --target-lang en  --trainpref $TEXT/train --validpref $TEXT/valid \
--testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
--tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25 --align-suffix align \


#===================Use the Pruned model 2023/08/14 regenerate the dataset to get correct test set Note: use the corrected test dataset =================
root=/home/valex/Documents/Study/battleship
DISTALL_DATA_PATH=$root/Reorder_nat/data/nat_position_reorder/awesome/baseline_detoken_distill_iwslt14_de_en
TOKEN_PATH=$root/Reorder_nat/data/nat_position_reorder/awesome/iwslt14_de_en_BlDist_dmbert
MODEL_NAME=distilbert-base-multilingual-cased
mkdir $TOKEN_PATH

for prefix in "valid" "test" "train" ;
do
    for lang in "en" "de" ;
    do
        echo "${prefix}.${lang} is processing"
        python transform_tokenize.py --input $DISTALL_DATA_PATH/${prefix}.${lang} --output $TOKEN_PATH/${prefix}.${lang} --pretrained_model $MODEL_NAME
    done
done


wget -O $TOKEN_PATH/src_vocab.txt https://huggingface.co/distilbert-base-multilingual-cased/resolve/main/vocab.txt
cp $TOKEN_PATH/src_vocab.txt $TOKEN_PATH/tgt_vocab.txt

# source $HOME/.bashrc 
# conda activate bibert

TEXT=$TOKEN_PATH

fairseq-preprocess --source-lang de --target-lang en  --trainpref $TEXT/train --validpref $TEXT/valid \
--testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
--tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25 --align-suffix align \