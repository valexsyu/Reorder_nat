# BIBERT_DATA=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/bibert/wmt14_detoken_awesome-align
# TRAIN_FILE=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_detoken_wmt14_de_en
# mkdir -p $TRAIN_FILE
# SRC=de
# TGT=en
# for prefix in "valid" "test" "train" ;
# do    
#     cp $BIBERT_DATA/$prefix.$SRC $TRAIN_FILE/$prefix.$SRC
#     cp $BIBERT_DATA/$prefix.$TGT $TRAIN_FILE/$prefix.$TGT
#     paste -d '  |||' $BIBERT_DATA/$prefix.$SRC /dev/null /dev/null /dev/null /dev/null /dev/null $BIBERT_DATA/$prefix.$TGT > $TRAIN_FILE/$prefix.paste.$SRC-$TGT

# done



# BIBERT_DISTILL_DATA=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/bibert/wmt14_detoken_distill_awesome-align
# TRAIN_DISTLL_FILE=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_detoken_distill_wmt14_de_en
# mkdir -p $TRAIN_DISTLL_FILE
# for prefix in "valid" "test" "train" ;
# do    
#     cp $BIBERT_DISTILL_DATA/$prefix.$SRC $TRAIN_DISTLL_FILE/$prefix.$SRC
#     cp $BIBERT_DISTILL_DATA/$prefix.$TGT $TRAIN_DISTLL_FILE/$prefix.$TGT
#     paste -d '  |||' $BIBERT_DISTILL_DATA/$prefix.$SRC /dev/null /dev/null /dev/null /dev/null /dev/null $BIBERT_DISTILL_DATA/$prefix.$TGT > $TRAIN_DISTLL_FILE/$prefix.paste.$SRC-$TGT

# done

DATAPATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/bibert/wmt20_jaen/non-distilled
TRAIN_FILE=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/wmt20_jaen_detoken_non-distilled
mkdir -p $TRAIN_FILE
SRC=ja
TGT=en
for prefix in "valid" "test" "train" ;
do    
    cp $DATAPATH/$prefix.$SRC $TRAIN_FILE/$prefix.$SRC
    cp $DATAPATH/$prefix.$TGT $TRAIN_FILE/$prefix.$TGT
    paste -d '  |||' $DATAPATH/$prefix.$SRC /dev/null /dev/null /dev/null /dev/null /dev/null $DATAPATH/$prefix.$TGT > $TRAIN_FILE/$prefix.paste.$SRC-$TGT

done