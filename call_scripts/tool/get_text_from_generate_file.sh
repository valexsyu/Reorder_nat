#!/bin/bash
#S:Sorce / T:Target / H:hypo / D:Detokenize

# how to use it ?
#1. cd to the generate-test.txt folder
#2. bash ../../../../call_scripts/tool/get_text_from_generate_file.sh

token_path=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/iwslt14_de_en_bibertDist_mbert_pruned26458
SRC=de
TGT=en

for sen_type in "T" "D" "H";
do
    if [ "$sen_type" = "T" ]
    then
        prefiex=target_word 
        cut_num=2
    elif [ "$sen_type" = "D" ]
    then    
        prefiex=hypo_word
        cut_num=3
    elif [ "$sen_type" = "H" ]
    then    
        prefiex=hypo_token
        cut_num=3        
    else
        "Error sen_type"
        exit 1
    fi

    ## get word number in each sentence 
    cat generate-test.txt  | head -n -2 | grep -P "^$sen_type" |sort -V |cut -f ${cut_num}- | awk '{print NF}' > ${prefiex}_num.txt
    ## get the sentence
    cat generate-test.txt  | head -n -2 | grep -P "^$sen_type" |sort -V |cut -f ${cut_num}-  > ${prefiex}_sentence.txt
done

cat $token_path/test.$TGT | awk '{print NF}' > target_token_num.txt
cat $token_path/test.$SRC | awk '{print NF}' > source_token_num.txt
echo "Done"