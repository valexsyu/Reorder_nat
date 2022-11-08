# FOLDER_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome
# #mkdir $FOLDER_PATH/Bibert_detoken_wmt14_de_en/complete_align
# DATA_INPUT=$FOLDER_PATH/Bibert_detoken_wmt14_de_en/
# DATA_OUTPUT=$FOLDER_PATH/Bibert_detoken_wmt14_de_en/
# SRC=de
# TGT=en
# #mkdir -p $FOLDER_PATH/Bibert_tokenize_wmt14_de_en/complete_align
# DATA_TOKENIZE_INPUT=$FOLDER_PATH/Bibert_tokenize_wmt14_de_en/
# DATA_TOKENIZE_OUTPUT=$FOLDER_PATH/Bibert_tokenize_wmt14_de_en/
# rsync -av --progress /home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/bibert/wmt-data/* \
#                      $DATA_TOKENIZE_OUTPUT \
#                      --exclude de-en-databin

# python complete_align.py --data-dir $DATA_INPUT --output-dir $DATA_OUTPUT \
#                          --data-token-dir $DATA_TOKENIZE_INPUT --output-token-dir $DATA_TOKENIZE_OUTPUT \
#                          --src $SRC --tgt $TGT 


# FOLDER_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome
# DATA_INPUT=$FOLDER_PATH/Bibert_detoken_distill_wmt14_de_en/
# DATA_OUTPUT=$FOLDER_PATH/Bibert_detoken_distill_wmt14_de_en/
# SRC=de
# TGT=en
# DATA_TOKENIZE_INPUT=$FOLDER_PATH/Bibert_token_distill_wmt14_de_en/
# DATA_TOKENIZE_OUTPUT=$FOLDER_PATH/Bibert_token_distill_wmt14_de_en/
# mkdir DATA_TOKENIZE_INPUT
# rsync -av --progress /home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/bibert/wmt14_detoken_distill_awesome-align/data/* \
#                      $DATA_TOKENIZE_OUTPUT \
#                      --exclude de-en-databin

# python complete_align.py --data-dir $DATA_INPUT --output-dir $DATA_OUTPUT \
#                          --data-token-dir $DATA_TOKENIZE_INPUT --output-token-dir $DATA_TOKENIZE_OUTPUT \
#                          --src $SRC --tgt $TGT       
                         
                                            
# FOLDER_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome
# DATA_INPUT=$FOLDER_PATH/Bibert_detoken_distill_wmt14_de_en/
# DATA_OUTPUT=$FOLDER_PATH/Bibert_detoken_distill_wmt14_de_en/
# SRC=de
# TGT=en
# DATA_TOKENIZE_INPUT=$FOLDER_PATH/Bibert_token_distill_wmt14_de_en_52k/
# DATA_TOKENIZE_OUTPUT=$FOLDER_PATH/Bibert_token_distill_wmt14_de_en_52k/
# mkdir DATA_TOKENIZE_INPUT
# rsync -av --progress /home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/bibert/wmt14_detoken_distill_awesome-align/bibert_tok/* \
#                      $DATA_TOKENIZE_OUTPUT \
#                      --exclude de-en-databin

# python complete_align.py --data-dir $DATA_INPUT --output-dir $DATA_OUTPUT \
#                          --data-token-dir $DATA_TOKENIZE_INPUT --output-token-dir $DATA_TOKENIZE_OUTPUT \
#                          --src $SRC --tgt $TGT     

# ##---------------------mbert------------------------------
# FOLDER_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome
# DATA_INPUT=$FOLDER_PATH/Bibert_detoken_distill_wmt14_de_en/
# DATA_OUTPUT=$FOLDER_PATH/Bibert_detoken_distill_wmt14_de_en/
# SRC=de
# TGT=en
# DATA_TOKENIZE_INPUT=$FOLDER_PATH/Bibert_token_distill_wmt14_de_en_mbert/
# DATA_TOKENIZE_OUTPUT=$FOLDER_PATH/Bibert_token_distill_wmt14_de_en_mbert/
# mkdir DATA_TOKENIZE_INPUT

# python complete_align.py --data-dir $DATA_INPUT --output-dir $DATA_OUTPUT \
#                          --data-token-dir $DATA_TOKENIZE_INPUT --output-token-dir $DATA_TOKENIZE_OUTPUT \
#                          --src $SRC --tgt $TGT                            

##---------------------dual-bert-de-en------------------------------
FOLDER_PATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome
DATA_INPUT=$FOLDER_PATH/Bibert_detoken_distill_wmt14_de_en/
DATA_OUTPUT=$FOLDER_PATH/Bibert_detoken_distill_wmt14_de_en/
SRC=de
TGT=en
DATA_TOKENIZE_INPUT=$FOLDER_PATH/Bibert_token_distill_wmt14_de_en_dual-bert-de-en/
DATA_TOKENIZE_OUTPUT=$FOLDER_PATH/Bibert_token_distill_wmt14_de_en_dual-bert-de-en/
mkdir DATA_TOKENIZE_INPUT

python complete_align.py --data-dir $DATA_INPUT --output-dir $DATA_OUTPUT \
                         --data-token-dir $DATA_TOKENIZE_INPUT --output-token-dir $DATA_TOKENIZE_OUTPUT \
                         --src $SRC --tgt $TGT        