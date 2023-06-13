
#!/bin/bash
# get_sentence -d file2.0.txt -d file3.0.txt -d file4.0.txt -t file2.0.txt -p c
# e.g. bash call_scripts/tool/cal_diff_score_to_get_max_bleu/get_sentence.sh \
#       -d generate-test-2.0.txt -d generate-test-3.0.txt -d generate-test-4.0.txt \
#       -t generate-test-2.0.txt -p m-B-3-1-N-UR30M-rate_avg/test/best_top5_10_1.bleu

root_path=/home/valexsyu/Doc/NMT/Reorder_nat/
# root_path=/home/valex/Documents/Study/battleship/Reorder_nat/checkpoints/
folder_path=sel_rate/m-B-1-1-N-UR20M-rate_sel-5k-rate_2_3_4/



VALID_ARGS=$(getopt -o d:t:p: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1;
fi

eval set -- "$VALID_ARGS"

while [ : ]; do

  case "$1" in 
    -d )
        data_name="$2"     
        data_name_array+=("$data_name")
        rate=$(echo "$data_name" | grep -oP '\d+\.\d+')
        rate_array+=("$rate")        
        shift 2
        ;;
    -t )
        tgt_name="$2"     
        shift 2
        ;;        
    -p )
        folder_path="$2"     
        shift 2
        ;;               
    --) shift; 
        break    
    esac
done

echo "========================================================"

file_path=$root_path$folder_path


if [ "${#data_name_array[@]}" -gt 0 ]; then
    echo "List of Bleu:"
    for i in "${data_name_array[@]}"; do
        # Do what you need based on $i
        # echo -e "\t$i"
        cat $file_path/$i | head -n -2 | grep -P "^D" |sort -V |cut -f 2-  > $file_path/D-$i
        parsing_data_array+=("$file_path/D-$i")

        bleu=$(cat $file_path/$i | tail -n -1 | cut -d ' ' -f 5-7 )
        echo "$i : $bleu "
    done
fi
cat $file_path/$tgt_name | head -n -2 | grep -P "^T" |sort -V |cut -f 2-  > $file_path/T-$tgt_name
parsing_tgt_array+=("$file_path/T-$tgt_name")


# echo "${data_name_array[@]}"
python call_scripts/tool/cal_diff_score_to_get_max_bleu/get_max_score_sentence.py --data ${parsing_data_array[@]} \
                           --tgt ${parsing_tgt_array[@]} \
                           --output_max_sentence $file_path/max_sentence.txt \
                           --output_index $file_path/index.txt
echo "========================================================"
perl call_scripts/tool/cal_diff_score_to_get_max_bleu/multi-bleu.perl $file_path/T-$tgt_name < $file_path/max_sentence.txt 


 