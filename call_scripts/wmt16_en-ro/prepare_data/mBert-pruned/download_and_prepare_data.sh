#example: cd download_and_prepare_data.sh
# bash download_and_prepare_data.sh
source $HOME/.bashrc 
conda activate base

SRC=en
TGT=ro
#===================Use the Pruned model 2023/06/11  Baseline distilled dataset =================
DISTALL_DATA_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/baseline_detoken_distilled_en-ro/demose
TOKEN_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/wmt16_en_ro_BlDist_cased_mbert_pruned29287
MODEL_NAME=/home/valexsyu/Doc/NMT/Reorder_nat/data/pruned_model/mBert/pruned_models/wmt16enro_pruned_V29287
## tokenize translation data
mkdir $TOKEN_PATH
cd /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/wmt14_en_de/prepare_data/mBert
for prefix in "valid" "test" "train" ; 
do
    for lang in "$SRC" "$TGT" ;
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
fairseq-preprocess --source-lang $SRC --target-lang $TGT  --trainpref $TOKEN_PATH/train --validpref $TOKEN_PATH/valid \
--testpref $TOKEN_PATH/test --destdir ${TOKEN_PATH}/de-en-databin --srcdict $TOKEN_PATH/src_vocab.txt \
--tgtdict $TOKEN_PATH/tgt_vocab.txt --vocab_file $TOKEN_PATH/src_vocab.txt --workers 25 --align-suffix align \