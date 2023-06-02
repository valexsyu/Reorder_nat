# e.g. 
# bash call_scripts/tool/cal_diff_score_to_get_max_bleu/Find_upper_bound/script.sh \
#       -d generate-test-2.0.txt -d generate-test-3.0.txt -d generate-test-4.0.txt \
#       -t generate-test-2.0.txt -p sel_rate/m-B-1-1-N-URXXM
root_path=/home/valexsyu/Doc/NMT/Reorder_nat/checkpoints/
folder_path=sel_rate/m-B-1-1-N-URXXM/

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
cat $file_path/$tgt_name | head -n -2 | grep -P "^T" |sort -V |cut -f 2-  > $file_path/T-$tgt_name

if [ "${#data_name_array[@]}" -gt 0 ]; then

    echo "List of Bleu:"
    for i in "${data_name_array[@]}"; do
        cat "$file_path/$i" | head -n -2 | grep -P "^D" | sort -V | cut -f 3  > "$file_path/D-$i"
        parsing_data_array+=("$file_path/D-$i")

        bleu=$(cat "$file_path/$i" | tail -n -1 | cut -d ' ' -f 5-7)
        echo "$i : $bleu "

        # output_file="$file_path/bleu_scores_$i.txt"  # Specify the output file path
        # python call_scripts/tool/cal_diff_score_to_get_max_bleu/Find_upper_bound/sentence_bleu.py \
        #             "$file_path/T-$tgt_name" "$file_path/D-$i" "$output_file"
    done
    
    output_bleu_file="$file_path/bleu_scores.txt"  # Specify the output file path
    output_index_file="$file_path/max_index.txt"  # Specify the output file path
    python call_scripts/tool/cal_diff_score_to_get_max_bleu/Find_upper_bound/sentence_bleu.py \
                --ref-path $file_path/T-$tgt_name \
                --hypo-path ${parsing_data_array[@]} \
                --output-bleu-path $output_bleu_file \
                --output-index-path $output_index_file \
                --output-bleu-fig-path "$file_path/bleu.png" \
                --output-index-fig-path "$file_path/index.png"
fi
