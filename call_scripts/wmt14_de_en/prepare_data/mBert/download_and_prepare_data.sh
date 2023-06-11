#example: cd download_and_prepare_data.sh
# bash download_and_prepare_data.sh
source $HOME/.bashrc 
conda activate base

# DISTALL_DATA_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_detoken_distill_wmt14_de_en
# TOKEN_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_token_distill_wmt14_de_en_mbert
# MODEL_NAME=bert-base-multilingual-uncased
# ## tokenize translation data
# mkdir $TOKEN_PATH
# cd /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/wmt14_de-en/prepare_data/mBert
# for prefix in "valid" "test" "train" ;
# do
#     for lang in "en" "de" ;
#     do
#         python transform_tokenize.py --input $DISTALL_DATA_PATH/${prefix}.${lang} --output $TOKEN_PATH/${prefix}.${lang} --pretrained_model $MODEL_NAME
#     done
# done

# ## get src and tgt vocabulary
# wget -O $TOKEN_PATH/src_vocab.txt https://huggingface.co/bert-base-multilingual-uncased/resolve/main/vocab.txt


# DISTALL_DATA_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/wmt14_clean_de_en_6kval_BigBlDist_cased_detoken
# TOKEN_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/wmt14_clean_de_en_6kval_BigBlDist_cased_mbert
# MODEL_NAME=bert-base-multilingual-cased
# ## tokenize translation data
# mkdir $TOKEN_PATH
# cd /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/wmt14_de_en/prepare_data/mBert
# for prefix in "valid" "test" "train" ; 
# do
#     for lang in "en" "de" ;
#     do
#         echo "${prefix}.${lang} tokenizing"
#         python transform_tokenize.py --input $DISTALL_DATA_PATH/${prefix}.${lang} --output $TOKEN_PATH/${prefix}.${lang} --pretrained_model $MODEL_NAME
#     done
# done

# ## get src and tgt vocabulary
# wget -O $TOKEN_PATH/src_vocab.txt https://huggingface.co/$MODEL_NAME/resolve/main/vocab.txt
# cp $TOKEN_PATH/src_vocab.txt $TOKEN_PATH/tgt_vocab.txt


#===================Use the Pruned model 2023/06/11 =================
DISTALL_DATA_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/wmt14_clean_de_en_6kval_BlDist_cased_detoken
TOKEN_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/wmt14_clean_de_en_6kval_BlDist_cased_mbert_pruned57959
MODEL_NAME=/home/valexsyu/Doc/NMT/Reorder_nat/data/pruned_model/mBert/pruned_models/wmt14deen_pruned_V57959
## tokenize translation data
mkdir $TOKEN_PATH
cd /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/wmt14_de_en/prepare_data/mBert
for prefix in "valid" "test" "train" ; 
do
    for lang in "en" "de" ;
    do
        echo "${prefix}.${lang} tokenizing"
        python transform_tokenize.py --input $DISTALL_DATA_PATH/${prefix}.${lang} --output $TOKEN_PATH/${prefix}.${lang} --pretrained_model $MODEL_NAME
    done
done

## get src and tgt vocabulary
cp $MODEL_NAME/vocab.txt \
   $TOKEN_PATH/src_vocab.txt
cp $MODEL_NAME/vocab.txt \
   $TOKEN_PATH/tgt_vocab.txt 

source $HOME/.bashrc 
conda activate bibert
fairseq-preprocess --source-lang de --target-lang en  --trainpref $TOKEN_PATH/train --validpref $TOKEN_PATH/valid \
--testpref $TOKEN_PATH/test --destdir ${TOKEN_PATH}/de-en-databin --srcdict $TOKEN_PATH/src_vocab.txt \
--tgtdict $TOKEN_PATH/tgt_vocab.txt --vocab_file $TOKEN_PATH/src_vocab.txt --workers 25 --align-suffix align \
