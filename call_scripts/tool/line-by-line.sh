source $HOME/.bashrc 
conda activate base

CHECKPOINTS_PATH=checkpoints

function get_dataset() {
    i=$(echo $1 | cut -d - -f 1)
    if [ "$i" = "1" ]     
    then
        dataset="iwslt14_de_en_bibertDist_mbert"
        SRC=de
    elif [ "$i" = "2" ]
    then
        dataset="iwslt14_de_en_bibertDist_bibert"
        SRC=de
    elif [ "$i" = "3" ]
    then
        dataset="iwslt14_de_en_BlDist_mbert"
        SRC=de
    elif [ "$i" = "4" ]
    then
        dataset="iwslt14_de_en_BlDist_bibert"
        SRC=de
    elif [ "$i" = "5" ]
    then
        dataset="iwslt14_de_en_bibertDist_dmbert"
        SRC=de
    elif [ "$i" = "6" ]
    then
        dataset="iwslt14_de_en_BlDist_dmbert"
        SRC=de
    elif [ "$i" = "7" ]
    then
        dataset="iwslt14_de_en_bibertDist_xlmr"
        SRC=de
    elif [ "$i" = "8" ]
    then
        dataset="iwslt14_de_en_BlDist_xlmr"                                
        SRC=de
    elif [ "$i" = "A" ]
    then
        dataset="wmt16_en_ro_BlDist_mbert"     
        SRC=en                           
    elif [ "$i" = "B" ]
    then
        dataset="wmt16_ro_en_BlDist_mbert" 
        SRC=ro                   
    elif [ "$i" = "C" ]
    then
        dataset="wmt16_ro_en_fncDist_mbert"         
        SRC=ro
    elif [ "$i" = "D" ]
    then
        dataset="wmt16_en_ro_mbartDist_mbert"   
        SRC=en
    elif [ "$i" = "E" ]
    then
        dataset="wmt14_en_de_bibertDist_bibert"     
        SRC=en
    elif [ "$i" = "F" ]
    then
        dataset="wmt14_de_en_bibertDist_bibert" 
        SRC=de
    elif [ "$i" = "G" ]
    then
        dataset="wmt14_en_de_bibert" 
        SRC=en
    elif [ "$i" = "H" ]
    then
        dataset="wmt14_de_en_bibert" 
        SRC=de
    elif [ "$i" = "I" ]
    then
        dataset="iwslt14_en_de_bibert" 
        SRC=en
    elif [ "$i" = "J" ]
    then
        dataset="iwslt14_de_en_bibert" 
        SRC=de
    elif [ "$i" = "K" ]
    then
        dataset="iwslt14_en_de_bibertDist_bibert"    
        SRC=en
    elif [ "$i" = "L" ]
    then
        dataset="wmt16_ro_en_mbert"                                                                                        
        SRC=ro
    elif [ "$i" = "M" ]
    then
        dataset="wmt16_en_ro_mbert"     
        SRC=en
    elif [ "$i" = "N" ]
    then
        dataset="wmt14_en_de_BlDist_bibert"    
        SRC=en
    elif [ "$i" = "O" ]
    then
        dataset="wmt14_de_en_BlDist_bibert"      
        SRC=de
    elif [ "$i" = "P" ]
    then
        dataset="iwslt14_en_de_BlDist_bibert"   
        SRC=en
    elif [ "$i" = "Q" ]
    then
        dataset="wmt14_en_de_BigBlDist_bibert"  
        SRC=en
    elif [ "$i" = "R" ]
    then
        dataset="wmt14_clean_en_de_bibert"   
        SRC=en
    elif [ "$i" = "S" ]
    then
        dataset="wmt14_clean_de_en_bibert"      
        SRC=de
    elif [ "$i" = "T" ]
    then
        dataset="wmt14_clean_de_en_bibertDist_bibert"      
        SRC=de
    elif [ "$i" = "U" ]
    then
        dataset="wmt14_clean_en_de_bibertDist_bibert" 
        SRC=en
    else        
        echo "error dataset id "
        exit 1
    fi
}


function default_setting() {
    twcc=False
    sleep_time=3600
    avg_speed=1
    no_atten_postfix=""
    data_subset=("test")
    CHECKPOINTS_PATH=checkpoints
    TOPK=5
    ck_types=("best_top$TOPK")
    BOOL_COMMAND=""
    no_atten_mask=False
    batch_size=60
    force=False
    dataroot=/livingrooms/valexsyu/dataset/nat
}


default_setting

VALID_ARGS=$(getopt -o e: --long experiment:,twcc,sleep:,avg-speed:,no-atten-mask,data-subset:,batch-size:,force,twcc -- "$@")
if [[ $? -ne 0 ]]; then
    echo "20"
    exit 1;
fi

eval set -- "$VALID_ARGS"

while [ : ]; do
  case "$1" in 
    -e | --experiment)
        experiment_ids="$2"     
        echo "$experiment_ids" 
        exp_array+=("$experiment_ids")
        shift 2
        ;;
    --twcc)
      twcc=True
      shift 1
      ;;   
    --force)
      force=True
      shift 1
      ;;           
    --avg-speed)
      avg_speed="$2"
      shift 2
      ;;          
    --sleep)
      sleep_time=$2
      shift 2
      ;;       
    --no-atten-mask)
      no_atten_mask=True
      no_atten_postfix="_NonMask"
      shift 1
      ;;    
    -b | --batch-size)
      batch_size="$2"
      shift 2
      ;;  
    --twcc)
      dataroot="../nat_data"
      shift 1
      ;;             
    --data-subset)
        case $2 in 
            test)
                data_subset=("test")
                ;;
            test-valid)
                data_subset=("test")
                data_subset+=("valid")
                ;;
            test-valid-train)
                data_subset=("test")
                data_subset+=("valid")
                data_subset+=("train")
                ;;
            valid)
                data_subset=("valid")
                ;;
            train)
                data_subset=("train")
                ;;                                
            *) 
                echo "data-setset id is wrong"
                exit 1    
        esac
      shift 2
      ;;                             
    --) shift; 
        break
  esac
done



if [ "$no_atten_mask" = "True" ]
then
      BOOL_COMMAND+=" --no-atten-mask "
      echo "$BOOL_COMMAND"
fi   


function sentence_bleu() {

    sed "${1}d" $2 > ${ref_file_tmp}_$5
    sed "${1}d" $3 > ${hyp_file_tmp}_$5  
    tmp_score=$(perl ./call_scripts/tool/mosesdecoder/scripts/generic/multi-bleu-test.perl ${ref_file_tmp}_$5 < ${hyp_file_tmp}_$5 )
    sub_tmp=$(echo $6 - $tmp_score | bc -l)
    echo "$1 , $(echo $sub_tmp / $6 | bc -l)" >> $RESULT_PATH/result.txt

}


for i in "${!exp_array[@]}"; do 
    experiment_id=${exp_array[$i]}
    CHECKPOINT=$CHECKPOINTS_PATH/$experiment_id
    get_dataset "$experiment_id"
    if [ -d "$CHECKPOINT" ]; then
        echo "=========No.$((i+1))  ID:$experiment_id:============="    
          for data_type in "${data_subset[@]}" ; do
              for ck_ch in "${ck_types[@]}"; do
                  checkpoint_data_name=checkpoint_$ck_ch.pt
                  if [ ! -f "$CHECKPOINT/$checkpoint_data_name" ]; then
                        echo "$checkpoint_data_name is not exist"
                        continue
                  fi  
                  echo -e "  data-subset: $data_type  ch-type: $ck_ch"
                  for (( speed_i=1; speed_i<=$avg_speed; speed_i++ )); do
                        RESULT_PATH=$CHECKPOINT/${data_type}$no_atten_postfix/${ck_ch}_${batch_size}_${speed_i}.bleu
                        FILE_PATH=$RESULT_PATH/generate-$data_type.txt

                        if [ ! -f "$FILE_PATH" ] || [ "$force" = "True" ]; then
                            echo "File Path is not exist, the checkpoint is $experiment_id"
                            bash call_scripts/generate_nat.sh -e $experiment_id $BOOL_COMMAND \
                                             --batch-size $batch_size --ck-types top --data-subset $data_type --avg-ck-turnoff 
                        else
                            lastln=$(tail -n1 $FILE_PATH)
                            last_generate_word=$(tail -n1 $FILE_PATH | awk '{print $1;}')
                            if [ "$last_generate_word" != "Generate" ]; then
                                echo "File Path is not exist, the checkpoint is $experiment_id"
                                bash call_scripts/generate_nat.sh -e $experiment_id $BOOL_COMMAND \
                                                --batch-size $batch_size --ck-types top --data-subset $data_type --avg-ck-turnoff 
                            fi
                        fi
                        
                        lastln=$(tail -n1 $FILE_PATH)
                        last_generate_word=$(tail -n1 $FILE_PATH | awk '{print $1;}')
                        if [ "$last_generate_word" = "Generate" ]
                        then
                              ref_file=$RESULT_PATH/ref
                              hyp_file=$RESULT_PATH/hyp
                              ref_file_tmp=$RESULT_PATH/ref_tmp
                              hyp_file_tmp=$RESULT_PATH/hyp_tmp
                              score_file_tmp=$RESULT_PATH/score_file
                              source_file=$dataroot/$dataset/$data_type.$SRC
                              cat $FILE_PATH | head -n -2 | grep -P "^T" |sort -V |cut -f 2- > $ref_file
                              cat $FILE_PATH | grep -P "^D" |sort -V |cut -f 3- > $hyp_file
                              original_score=$(perl ./call_scripts/tool/mosesdecoder/scripts/generic/multi-bleu-test.perl $ref_file < $hyp_file)
                              num_lines=$(cat $ref_file | wc -l)
                              
                            if [ -f $score_file_tmp ]; then
                                rm $score_file_tmp
                            fi
                            if [ -f $RESULT_PATH/result.txt ]; then
                                rm $RESULT_PATH/result.txt
                            fi                            
                            num_processes=100
                            for (( c=1; c<=$num_lines; c++ ))
                            do  
                                sed "${c}d" $ref_file > $ref_file_tmp  
                                sed "${c}d" $hyp_file > $hyp_file_tmp  
                                word_num=$(sed -n "${c}p"  $source_file | wc -w)
                                tmp_score=$(perl ./call_scripts/tool/mosesdecoder/scripts/generic/multi-bleu-test.perl $ref_file_tmp < $hyp_file_tmp)
                                sub_tmp=$(echo $original_score - $tmp_score | bc -l)
                                echo "$c ,$word_num, $(echo $sub_tmp / $original_score | bc -l)" >> $RESULT_PATH/result.txt                                

                                # ((i=i%num_processes)); ((i++==0)) && wait
                                # sentence_bleu $c $ref_file $hyp_file $score_file_tmp $i $original_score &
                                echo -ne "$c/$num_lines \r"
                                rm $ref_file_tmp 
                                rm $hyp_file_tmp 
                            done                              
                        else
                              bash call_scripts
                              echo " $FILE_PATH is Fail , Error code last_generate_word != Generate"     
                              exit 1           
                        fi
                  done
              done
              
          done
    else
        no_exp_array+=("$experiment_id")
    fi   
done

if [ "${#no_exp_array[@]}" -gt 0 ]; then
    echo "The experiments are NOT in the checkpoings path:"
    for i in "${no_exp_array[@]}"; do
        # Do what you need based on $i
        echo -e "\t$i"
    done
fi