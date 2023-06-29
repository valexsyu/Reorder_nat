# DATA_ROOT=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_token_distill_wmt14_de_en_52k
DATA_ROOT=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/iwslt14_de_en_bibertDist_mbert_pruned26458
LANGS=("de" "en")
DATA_TYPES=("test")
RESULT_ROOT=$DATA_ROOT/num_words_each_line/
mkdir $RESULT_ROOT
for type in "${DATA_TYPES[@]}" ; do
    for lg in "${LANGS[@]}" ; do
        DATA_PATH=$DATA_ROOT/$type.$lg
        RESULT_PATH=$RESULT_ROOT/num_$type.$lg
        awk '{print  NF}' $DATA_PATH > $RESULT_PATH   # output the number of line
    done
done

# output de and en number of line
for type in "${DATA_TYPES[@]}" ; do
    DATA1=$DATA_ROOT/$type.de
    DATA2=$DATA_ROOT/$type.en
    RESULT_PATH=$RESULT_ROOT/num_$type.txt
    awk 'FNR>=1 && NR==FNR {out[FNR]=NF ; next}
         FNR>=1 { printf "%d %d\n", out[FNR], NF }' $DATA1 $DATA2 > $RESULT_PATH
done

# python call_scripts/count_word_line.py --data-dir $RESULT_ROOT