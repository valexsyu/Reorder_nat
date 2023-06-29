# e.g. 
# bash call_scripts/tool/cal_diff_score_to_get_max_bleu/Find_upper_bound/script.sh \
#       -d generate-test-2.0.txt -d generate-test-3.0.txt -d generate-test-4.0.txt \
#       -t generate-test-2.0.txt -p sel_rate/m-B-1-1-N-URXXM -e m
# bash call_scripts/tool/cal_diff_score_to_get_max_bleu/Find_upper_bound/script.sh \
#       -d generate-test-2.0.txt -d generate-test-3.0.txt -d generate-test-4.0.txt \
#       -t generate-test-2.0.txt -p sel_rate/out.wmt14.de-en.alg1 -e Fail (out.wmt14.de-en No dataset)
# bash call_scripts/tool/cal_diff_score_to_get_max_bleu/Find_upper_bound/script.sh \
#       -d generate-test-2.0.txt -d generate-test-2.5.txt -d generate-test-3.0.txt \
#       -d generate-test-3.5.txt  -d generate-test-4.0.txt  -t generate-test-2.0.txt \
#       -p checkpoints/m-B-3-1-N-UR30M-rate_avg_1/test/best_top5_10_1.bleu/ -e m
function get_dataset() {
    i=$(echo $1 | cut -d - -f 1)
    if [ "$i" = "1" ]     
    then
        dataset="iwslt14_de_en_bibertDist_mbert"
        LANGS=("de" "en")
    elif [ "$i" = "2" ]
    then
        dataset="iwslt14_de_en_bibertDist_bibert"
        LANGS=("de" "en")
    elif [ "$i" = "3" ]
    then
        dataset="iwslt14_de_en_BlDist_mbert"
        LANGS=("de" "en")
    elif [ "$i" = "4" ]
    then
        dataset="iwslt14_de_en_BlDist_bibert"
        LANGS=("de" "en")
    elif [ "$i" = "5" ]
    then
        dataset="iwslt14_de_en_bibertDist_dmbert"
        LANGS=("de" "en")
    elif [ "$i" = "6" ]
    then
        dataset="iwslt14_de_en_BlDist_dmbert"
        LANGS=("de" "en")
    elif [ "$i" = "7" ]
    then
        dataset="iwslt14_de_en_bibertDist_xlmr"
        LANGS=("de" "en")
    elif [ "$i" = "8" ]
    then
        dataset="iwslt14_de_en_BlDist_xlmr"                                
        LANGS=("de" "en")
    elif [ "$i" = "A" ]
    then
        dataset="wmt16_en_ro_BlDist_mbert"                                
        LANGS=("en" "ro")
    elif [ "$i" = "B" ]
    then
        dataset="wmt16_ro_en_BlDist_mbert"   
        LANGS=("en" "ro")                 
    elif [ "$i" = "C" ]
    then
        dataset="wmt16_ro_en_fncDist_mbert"         
        LANGS=("ro" "en")
    elif [ "$i" = "D" ]
    then
        dataset="wmt16_en_ro_mbartDist_mbert"   
        LANGS=("en" "ro")
    elif [ "$i" = "E" ]
    then
        dataset="wmt14_en_de_bibertDist_bibert"     
        LANGS=("en" "de")
    elif [ "$i" = "F" ]
    then
        dataset="wmt14_de_en_bibertDist_bibert" 
        LANGS=("en" "de")
    elif [ "$i" = "G" ]
    then
        dataset="wmt14_en_de_bibert" 
        LANGS=("en" "de")
    elif [ "$i" = "H" ]
    then
        dataset="wmt14_de_en_bibert" 
        LANGS=("de" "en")
    elif [ "$i" = "I" ]
    then
        dataset="iwslt14_en_de_bibert" 
        LANGS=("en" "de")
    elif [ "$i" = "J" ]
    then
        dataset="iwslt14_de_en_bibert" 
        LANGS=("de" "en")
    elif [ "$i" = "K" ]
    then
        dataset="iwslt14_en_de_bibertDist_bibert"         
        LANGS=("en" "de")                                                                                   
    elif [ "$i" = "L" ]
    then
        dataset="wmt16_ro_en_mbert"                                                                                        
        LANGS=("ro" "en")
    elif [ "$i" = "M" ]
    then
        dataset="wmt16_en_ro_mbert"    
        LANGS=("en" "ro")
    elif [ "$i" = "N" ]
    then
        dataset="wmt14_en_de_BlDist_bibert"    
        LANGS=("en" "de")
    elif [ "$i" = "O" ]
    then
        dataset="wmt14_de_en_BlDist_bibert"      
        LANGS=("de" "en")
    elif [ "$i" = "P" ]
    then
        dataset="iwslt14_en_de_BlDist_bibert"   
    elif [ "$i" = "Q" ]
    then
        dataset="wmt14_en_de_BigBlDist_bibert"   
        LANGS=("en" "de")   
    elif [ "$i" = "R" ]
    then
        dataset="wmt14_clean_en_de_bibert"   
        LANGS=("en" "de")
    elif [ "$i" = "S" ]
    then
        dataset="wmt14_clean_de_en_bibert"    
        LANGS=("de" "en")
    elif [ "$i" = "T" ]
    then
        dataset="wmt14_clean_de_en_bibertDist_bibert"      
        LANGS=("de" "en")
    elif [ "$i" = "U" ]
    then
        dataset="wmt14_clean_en_de_bibertDist_bibert"  
        LANGS=("en" "de")
    elif [ "$i" = "V" ]
    then
        dataset="wmt16_en_ro_BigBlDist_mbert" 
        LANGS=("en" "ro")     
    elif [ "$i" = "W" ]
    then
        dataset="wmt16_ro_en_BigBlDist_mbert"
        LANGS=("ro" "en")
    elif [ "$i" = "X" ]
    then
        dataset="wmt14_de_en_BigBlDist_bibert"     
        LANGS=("de" "en")
    elif [ "$i" = "Y" ]
    then
        dataset="wmt14_clean_de_en_6kval_bibertDist_bibert"        
        LANGS=("de" "en")
    elif [ "$i" = "Z" ]
    then
        dataset="wmt14_clean_en_de_6kval_bibertDist_bibert"    
        LANGS=("en" "de")
    elif [ "$i" = "a" ]
    then
        dataset="wmt14_clean_de_en_6kval_bibert"        
        LANGS=("de" "en")
    elif [ "$i" = "b" ]
    then
        dataset="wmt14_clean_en_de_6kval_bibert"                                     
        LANGS=("en" "de")
    elif [ "$i" = "c" ]
    then
        dataset="wmt20_ja_en_BlDist_mbert"       
        LANGS=("ja" "en")
    elif [ "$i" = "d" ]
    then
        dataset="wmt20_ja_en_mbert"  
        LANGS=("ja" "en")  
    elif [ "$i" = "e" ]
    then
        dataset="wmt14_clean_en_de_6kval_BigBlDist_cased_mbert"     
        LANGS=("en" "de")
    elif [ "$i" = "f" ] 
    then
        dataset="wmt14_clean_en_de_6kval_BlDist_cased_mbert"   
        LANGS=("en" "de")
    elif [ "$i" = "g" ]
    then
        dataset="wmt14_clean_de_en_6kval_BigBlDist_cased_mbert"   
        LANGS=("de" "en")
    elif [ "$i" = "h" ]
    then
        dataset="wmt14_clean_de_en_6kval_BlDist_cased_mbert"        
        LANGS=("de" "en")
    elif [ "$i" = "i" ]
    then
        dataset="iwslt14_de_en_BlDist_cased_mbert"         
        LANGS=("de" "en")
    elif [ "$i" = "j" ]
    then
        dataset="wmt14_en_de_BigBlDist_xlmr"       
        LANGS=("en" "de")
    elif [ "$i" = "k" ]
    then
        dataset="iwslt14_de_en_BlDist_mbertcased"            
        LANGS=("de" "en")
    elif [ "$i" = "m" ]
    then
        dataset="iwslt14_de_en_bibertDist_mbert_pruned26458"            
        LANGS=("de" "en")
    elif [ "$i" = "n" ]
    then
        dataset="iwslt14_de_en_bibertDist_mbert_pruned26458_8k"
        LANGS=("de" "en")
    elif [ "$i" = "o" ]
    then
        dataset="iwslt14_de_en_bibertDist_xlmr_pruned21785"     
        LANGS=("de" "en")
    elif [ "$i" = "p" ]
    then
        dataset="iwslt14_de_en_bibertDist_bibert_pruned43093"        
        LANGS=("de" "en")
    elif [ "$i" = "q" ]
    then
        dataset="iwslt14_de_en_mbert_pruned26458"      
        LANGS=("de" "en")
    elif [ "$i" = "r" ]
    then
        dataset="wmt14_clean_de_en_6kval_BlDist_cased_mbert_pruned57959"  
        LANGS=("de" "en")
    elif [ "$i" = "s" ]
    then
        dataset="wmt14_clean_en_de_6kval_BlDist_cased_mbert_pruned58003"           
        LANGS=("en" "de")
    elif [ "$i" = "t" ]
    then
        dataset="wmt16_ro_en_BlDist_cased_mbert_pruned29271"   
        LANGS=("ro" "en") 
    elif [ "$i" = "u" ]
    then
        dataset="wmt16_en_ro_BlDist_cased_mbert_pruned29287"      
        LANGS=("en" "ro")
    elif [ "$i" = "v" ]
    then
        dataset="iwslt14_de_en_BlDist_mbert_Bl_pruned25958" 
        LANGS=("de" "en")                                    
    else      
        echo "error dataset id "
        echo $1
        exit 1
    fi
} 

function default_setting() {
    dataroot=/livingrooms/valexsyu/dataset/nat  
    twcc=False  
    root_path=/home/valexsyu/Doc/NMT/Reorder_nat/
    # root_path=/home/valex/Documents/Study/battleship/Reorder_nat/checkpoints/
    folder_path=sel_rate/m-B-1-1-N-UR20M-rate_sel-5k-rate_2_3_4/    
}


default_setting
# root_path=/home/valexsyu/Doc/NMT/Reorder_nat/
# folder_path=sel_rate/m-B-1-1-N-URXXM/

VALID_ARGS=$(getopt -o e:d:t:p: --long local -- "$@")
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
        -e | --experiment)
        experiment_id="$2"
        shift 2
        ;;  
        --local)
        dataroot="../../dataset/nat"
        shift 1
        ;;                   
        --) shift; 
            break    
    esac
done

echo "========================================================"

get_dataset $experiment_id
file_path=$root_path$folder_path

DATASET_PATH=$dataroot/$dataset

# =================================get sorce and target number of token=================
DATA_TYPES=("test")
for type in "${DATA_TYPES[@]}" ; do
    for lg in "${LANGS[@]}" ; do
        DATA_PATH=$DATASET_PATH/$type.$lg
        RESULT_PATH=$file_path/num_$type.$lg
        awk '{print  NF}' $DATA_PATH > $RESULT_PATH   # output the number of line
    done
done

cat $file_path/$tgt_name | head -n -2 | grep -P "^T" |sort -V |cut -f 2-  > $file_path/T-$tgt_name

if [ "${#data_name_array[@]}" -gt 0 ]; then

    echo "List of Bleu:"
    # for i in "${data_name_array[@]}"; do
    #     cat $file_path/$i | head -n -2 | grep -P "^D" | sort -V | cut -f 3-  > $file_path/D-$i
    #     parsing_data_array+=("$file_path/D-$i")

    #     bleu=$(cat "$file_path/$i" | tail -n -1 | cut -d ' ' -f 5-7)
    #     echo "$i : $bleu "

    #     # output_file="$file_path/bleu_scores_$i.txt"  # Specify the output file path
    #     # python call_scripts/tool/cal_diff_score_to_get_max_bleu/Find_upper_bound/sentence_bleu.py \
    #     #             "$file_path/T-$tgt_name" "$file_path/D-$i" "$output_file"
    # done
    


    for i in "${data_name_array[@]}"; do
        # Do what you need based on $i
        # echo -e "\t$i"
        cat $file_path/$i | head -n -2 | grep -P "^D" | sort -V | cut -f 2-  > $file_path/D-${i}-score
        parsing_data_array+=("$file_path/D-${i}-score")

        bleu=$(cat "$file_path/$i" | tail -n -1 | cut -d ' ' -f 5-7)
        echo "$i : $bleu "        

    done


    output_bleu_file="$file_path/bleu_scores_uperbound.txt"  # Specify the output file path
    output_index_file="$file_path/max_index_uperbound.txt"  # Specify the output file path
    python call_scripts/tool/cal_diff_score_to_get_max_bleu/Find_upper_bound/sentence_bleu.py \
                --ref-path $file_path/T-$tgt_name \
                --hypo-path ${parsing_data_array[@]} \
                --output-bleu-path $output_bleu_file \
                --output-index-path $output_index_file \
                --output-bleu-fig-path "$file_path/bleu_uperbound.png" \
                --output-index-fig-path "$file_path/index_uperbound.png" \
                --output-dir "$file_path" \
                --output-max-sentence-prob $file_path/max_sentence_prob.txt \
                --output-index-prob $file_path/index_prob.txt \
                --source-token-num $file_path/num_test.${LANGS[0]} \
                --target-token-num $file_path/num_test.${LANGS[1]}                

fi


# # =================================get sorce and target number of token=================

# DATA_TYPES=("test")
# for type in "${DATA_TYPES[@]}" ; do
#     for lg in "${LANGS[@]}" ; do
#         DATA_PATH=$DATASET_PATH/$type.$lg
#         RESULT_PATH=$file_path/num_$type.$lg
#         awk '{print  NF}' $DATA_PATH > $RESULT_PATH   # output the number of line
#     done
# done




# if [ "${#data_name_array[@]}" -gt 0 ]; then
#     echo "List of Bleu:"
#     for i in "${data_name_array[@]}"; do
#         # Do what you need based on $i
#         # echo -e "\t$i"
#         cat $file_path/$i | head -n -2 | grep -P "^D" |sort -V |cut -f 2-  > $file_path/D-$i
#         parsing_data_array+=("$file_path/D-$i")

#         bleu=$(cat $file_path/$i | tail -n -1 | cut -d ' ' -f 5-7 )
#         echo "$i : $bleu "
#     done
# fi
# cat $file_path/$tgt_name | head -n -2 | grep -P "^T" |sort -V |cut -f 2-  > $file_path/T-$tgt_name
# parsing_tgt_array+=("$file_path/T-$tgt_name")


# # echo "${data_name_array[@]}"
# python call_scripts/tool/cal_diff_score_to_get_max_bleu/get_max_score_sentence.py --data ${parsing_data_array[@]} \
#                            --tgt ${parsing_tgt_array[@]} \
#                            --output_dir $file_path \
#                            --output_max_sentence $file_path/max_sentence.txt \
#                            --output_index $file_path/index.txt \
#                            --source_token_num $file_path/num_test.${LANGS[0]} \
#                            --target_token_num $file_path/num_test.${LANGS[1]}
# echo "========================================================"
# perl call_scripts/tool/cal_diff_score_to_get_max_bleu/multi-bleu.perl $file_path/T-$tgt_name < $file_path/max_sentence.txt 
