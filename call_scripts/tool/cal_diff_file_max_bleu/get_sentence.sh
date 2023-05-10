
#!/bin/bash

root_path=/home/valex/Documents/Study/battleship/Reorder_nat/checkpoints/
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

file_paht=$root_path$folder_path


if [ "${#data_name_array[@]}" -gt 0 ]; then
    echo "List of Bleu:"
    for i in "${data_name_array[@]}"; do
        # Do what you need based on $i
        # echo -e "\t$i"
        cat $file_paht/$i | head -n -2 | grep -P "^D" |sort -V |cut -f 2-  > $file_paht/D-$i
        parsing_data_array+=("$file_paht/D-$i")

        bleu=$(cat $file_paht/$i | tail -n -1 | cut -d ' ' -f 5-7 )
        echo "$i : $bleu "
    done
fi
cat $file_paht/$tgt_name | head -n -2 | grep -P "^T" |sort -V |cut -f 2-  > $file_paht/T-$tgt_name
parsing_tgt_array+=("$file_paht/T-$tgt_name")


# echo "${data_name_array[@]}"
python get_max_sentence.py --data ${parsing_data_array[@]} \
                           --tgt ${parsing_tgt_array[@]} \
                           --output_max_sentence $file_paht/max_sentence.txt \
                           --output_index $file_paht/index.txt
echo "========================================================"
echo "Using maximu rate:"
perl multi-bleu.perl $file_paht/T-$tgt_name < $file_paht/max_sentence.txt  