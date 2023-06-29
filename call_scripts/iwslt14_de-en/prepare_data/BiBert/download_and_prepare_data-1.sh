source $HOME/.bashrc 
conda activate bibert
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

# DISTALL_DATA_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_detoken_iwslt14_de_en
# TOKEN_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_token_iwslt14_de_en_mbert
# MODEL_NAME_EN=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_token_iwslt14_de_en_mbert/tgt_vocab.txt
# MODEL_NAME_DE=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_token_iwslt14_de_en_mbert/src_vocab.txt
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

#===================Use the Pruned model 2023/06/27 Bibert with the Bibert distilled dataset Note: use the corrected test dataset =================
DISTALL_DATA_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_detoken_distill_iwslt14_de_en
TOKEN_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_token_distill_iwslt14_de_en_bibert
PRUN_MODEL_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_token_distill_iwslt14_de_en_bibert/src_vocab.txt
mkdir $TOKEN_PATH

for prefix in "valid" "test" "train" ;
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

fairseq-preprocess --source-lang de --target-lang en  --trainpref $TEXT/train --validpref $TEXT/valid \
--testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
--tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25 --align-suffix align \
