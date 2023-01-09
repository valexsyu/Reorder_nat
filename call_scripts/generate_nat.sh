#!/bin/bash
source $HOME/.bashrc 
conda activate base

#---------Path Setting-------------------#
# Model    Bibert Tr.   #Baseline Tr.
# mbert       1            3
# bibert      2            4 
# dmbert      5            6
# xlmr        7            8

function get_dataset() {
    i=$(echo $1 | cut -d - -f 1)
    if [ "$i" = "1" ]     
    then
        dataset="iwslt14_de_en_bibertDist_mbert"
    elif [ "$i" = "2" ]
    then
        dataset="iwslt14_de_en_bibertDist_bibert"
    elif [ "$i" = "3" ]
    then
        dataset="iwslt14_de_en_BlDist_mbert"
    elif [ "$i" = "4" ]
    then
        dataset="iwslt14_de_en_BlDist_bibert"
    elif [ "$i" = "5" ]
    then
        dataset="iwslt14_de_en_bibertDist_dmbert"
    elif [ "$i" = "6" ]
    then
        dataset="iwslt14_de_en_BlDist_dmbert"
    elif [ "$i" = "7" ]
    then
        dataset="iwslt14_de_en_bibertDist_xlmr"
    elif [ "$i" = "8" ]
    then
        dataset="iwslt14_de_en_BlDist_xlmr"                                
    elif [ "$i" = "A" ]
    then
        dataset="wmt16_en_ro_BlDist_mbert"                                
    elif [ "$i" = "B" ]
    then
        dataset="wmt16_ro_en_BlDist_mbert"                    
    elif [ "$i" = "C" ]
    then
        dataset="wmt16_ro_en_fncDist_mbert"         
    elif [ "$i" = "D" ]
    then
        dataset="wmt16_en_ro_mbartDist_mbert"   
    elif [ "$i" = "E" ]
    then
        dataset="wmt14_en_de_bibertDist_bibert"     
    elif [ "$i" = "F" ]
    then
        dataset="wmt14_de_en_bibertDist_bibert" 
    elif [ "$i" = "G" ]
    then
        dataset="wmt14_en_de_bibert" 
    elif [ "$i" = "H" ]
    then
        dataset="wmt14_de_en_bibert" 
    elif [ "$i" = "I" ]
    then
        dataset="iwslt14_en_de_bibert" 
    elif [ "$i" = "J" ]
    then
        dataset="iwslt14_de_en_bibert" 
    elif [ "$i" = "K" ]
    then
        dataset="iwslt14_en_de_bibertDist_bibert"    
    elif [ "$i" = "L" ]
    then
        dataset="wmt16_ro_en_mbert"                                                                                        
    elif [ "$i" = "M" ]
    then
        dataset="wmt16_en_ro_mbert"     
    elif [ "$i" = "N" ]
    then
        dataset="wmt14_en_de_BlDist_bibert"    
    elif [ "$i" = "O" ]
    then
        dataset="wmt14_de_en_BlDist_bibert"      
    elif [ "$i" = "P" ]
    then
        dataset="iwslt14_en_de_BlDist_bibert"   
    elif [ "$i" = "Q" ]
    then
        dataset="wmt14_en_de_BigBlDist_bibert"  
    elif [ "$i" = "R" ]
    then
        dataset="wmt14_clean_en_de_bibert"   
    elif [ "$i" = "S" ]
    then
        dataset="wmt14_clean_de_en_bibert"      
    elif [ "$i" = "T" ]
    then
        dataset="wmt14_clean_de_en_bibertDist_bibert"      
    elif [ "$i" = "U" ]
    then
        dataset="wmt14_clean_en_de_bibertDist_bibert" 
    elif [ "$i" = "V" ]
    then
        dataset="wmt16_en_ro_BigBlDist_mbert"      
    elif [ "$i" = "W" ]
    then
        dataset="wmt16_ro_en_BigBlDist_mbert"  
    elif [ "$i" = "X" ]
    then
        dataset="wmt14_de_en_BigBlDist_bibert"  
    elif [ "$i" = "Y" ]
    then
        dataset="wmt14_clean_de_en_6kval_bibertDist_bibert"        
    elif [ "$i" = "Z" ]
    then
        dataset="wmt14_clean_en_de_6kval_bibertDist_bibert"    
    elif [ "$i" = "a" ]
    then
        dataset="wmt14_clean_de_en_6kval_bibert"        
    elif [ "$i" = "b" ]
    then
        dataset="wmt14_clean_en_de_6kval_bibert"       
    elif [ "$i" = "c" ]
    then
        dataset="wmt20_ja_en_BlDist_mbert"       
    elif [ "$i" = "d" ]
    then
        dataset="wmt20_ja_en_mbert"    
    elif [ "$i" = "e" ]
    then
        dataset="wmt14_clean_en_de_6kval_BigBlDist_cased_mbert"  
    elif [ "$i" = "f" ]
    then
        dataset="wmt14_clean_en_de_6kval_BlDist_cased_mbert"   
    elif [ "$i" = "g" ]
    then
        dataset="wmt14_clean_de_en_6kval_BigBlDist_cased_mbert"   
    elif [ "$i" = "h" ]
    then
        dataset="wmt14_clean_de_en_6kval_BlDist_cased_mbert"                                            
    else        
        echo "error dataset id "
        exit 1
    fi
}

function get_pretrain_model() {
    i=$(echo $1 | cut -d - -f 2)
    if [ "$i" = "1" ]
    then
        pretrained_model="mbert"
        pretrained_model_name="bert-base-multilingual-uncased"
        bpe="bibert"
    elif [ "$i" = "2" ]
    then
        pretrained_model="bibert"
        pretrained_model_name="jhu-clsp/bibert-ende"
        bpe="bibert"
    elif [ "$i" = "3" ]
    then
        pretrained_model="dmbert"
        pretrained_model_name="distilbert-base-multilingual-cased"
        bpe="bibert"
    elif [ "$i" = "4" ]
    then
        pretrained_model="xlmr"
        pretrained_model_name="xlm-roberta-base"
        bpe="xlmr"
    elif [ "$i" = "5" ]
    then
        pretrained_model="mbert"
        pretrained_model_name="bert-base-multilingual-uncased"     
        bpe="bibert"
    elif [ "$i" = "6" ]
    then
        pretrained_model="bibert"
        pretrained_model_name="jhu-clsp/bibert-ende"
        bpe="bibert"    
    elif [ "$i" = "7" ]
    then
        pretrained_model="mbert-cased"
        pretrained_model_name="bert-base-multilingual-cased"
        bpe="bibert"                 
    else
        echo "error pretrained model id "
        exit 1
    fi
}

function get_fix_lm_swe() {
    i=$(echo $1 | cut -d - -f 3)
    if [ "$i" = "1" ]
    then
        fix_lm=True
        fix_swe=True
    elif [ "$i" = "2" ]
    then
        fix_lm=True
        fix_swe=False
    elif [ "$i" = "3" ]
    then
        fix_lm=False
        fix_swe=True
    elif [ "$i" = "4" ]
    then
        fix_lm=False
        fix_swe=False
    else
        echo "error fix lm and swe id "
        exit 1
    fi
}

function get_voc() {
    i=$(echo $1 | cut -d - -f 4)
    if [ "$i" = "1" ]
    then
        voc="1"
    elif [ "$i" = "2" ]
    then
        voc="2"
    elif [ "$i" = "3" ]
    then
        voc="3"
    else
        echo "error voc id "
        exit 1
    fi
}

function get_kd_model() { 
    i=$(echo $1 | cut -d - -f 5)
    if [ $(echo $i | cut -c 1) = "H" ]
    then
        lm_loss_layer=$(($(echo $i | cut -c 2-3)-13))
        lm_loss_dis=False
        lm_loss=True        
    else
        if [ "$i" = "T" ]
        then
            lm_loss_dis=True
            lm_loss_layer=-1
            lm_loss=True
        elif [ "$i" = "H" ]
        then
            lm_loss_dis=False
            lm_loss_layer=-1
            lm_loss=True
        elif [ "$i" = "N" ]
        then
            lm_loss_dis=False
            lm_loss_layer=-1
            lm_loss=False
        else
            echo "error kd model id "
            exit 1
        fi
    fi
}

function get_ctc() {
    i=$(echo $1 | cut -d - -f 6)
    case "$(echo $i | cut -c 1)" in 
        U)
            insert_position="uniform"
            ;;
        R)
            insert_position="right"
            ;;
        L)
            insert_position="left"
            ;;   
        *) 
            echo "insert position is wrong id"
            exit 1    
    esac
    case "$(echo $i | cut -c 2)" in 
        F)
            dynamic_upsampling=False
            ;;
        D)
            dynamic_upsampling=True
            ;;
        R)
            dynamic_rate=True
            ;;
        *) 
            echo "dynamic upsampling is wrong id"
            exit 1    
    esac

    num_upsampling_rate="$(echo $i | cut -c 3-4)"
    case "$(echo $i | cut -c 5)" in 
        M)
            insert_mask=True
            ;;
        T)
            insert_mask=False
            ;;
        *) 
            echo "insert_mask is wrong id"
            exit 1    
    esac         

}

function default_setting() {
    gpu=1
    batch_size=60
    max_tokens=2048
    max_epoch=400
    update_freq=6
    dataroot=/livingrooms/valexsyu/dataset/nat
    cpu=False
    data_subset=("test")
    TOPK=5
    ck_types=("last" "best" "best_top$TOPK" "last$TOPK")
    load_exist_bleu=False
    avg_ck_turnoff=False
    no_atten_mask=False
    skip_exist_genfile=False
    no_atten_postfix=""
    avg_speed=1
    local=False
    
}

function avg_topk_best_checkpoints(){
	python scripts/average_checkpoints.py \
	            --inputs $1 \
				--num-epoch-checkpoints $2 --output $3 \
				--ckpref checkpoints.best_bleu	
}

function avg_lastk_checkpoints(){
	python scripts/average_checkpoints.py \
	            --inputs $1 \
				--num-epoch-checkpoints $2 --output $3
}


default_setting

VALID_ARGS=$(getopt -o e:,b: --long experiment:,twcc,local,batch-size:,cpu,data-subset:,debug,load-exist-bleu,ck-types:,avg-ck-turnoff,no-atten-mask,skip-exist-genfile,avg-speed: -- "$@")
if [[ $? -ne 0 ]]; then
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
      dataroot="../nat_data"
      shift 1
      ;;     
    --local)
      dataroot="../../dataset/nat"
      local=True
      shift 1
      ;;       
    --cpu)
      cpu=True
      shift 1
      ;;      
    --avg-ck-turnoff)
      avg_ck_turnoff=True
      shift 1
      ;;          
    --debug)
      debug=True
      shift 1
      ;;  
    --load-exist-bleu)
      load_exist_bleu=True
      shift 1
      ;;       
    --no-atten-mask)
      no_atten_mask=True
      no_atten_postfix="_NonMask"
      shift 1
      ;;       
    --skip-exist-genfile)
      skip_exist_genfile=True
      shift 1
      ;;                             
    -b | --batch-size)
      batch_size="$2"
      shift 2
      ;;       
    --avg-speed)
      avg_speed="$2"
      shift 2
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
    --ck-types)
        case $2 in 
            top)
                ck_types=("best_top$TOPK")
                ;;
            top-lastk)
                ck_types=("best_top$TOPK")
                ck_types+=("last$TOPK")
                ;;                
            best-top)
                ck_types=("best")
                ck_types+=("best_top$TOPK")
                ;;
            last-top)
                ck_types=("last")
                ck_types+=("best_top$TOPK")
                ;; 
            last-top-lastk)
                ck_types=("last")
                ck_types+=("best_top$TOPK")
                ck_types+=("last$TOPK")
                ;;                                 
            last-best-top)
                ck_types=("last")
                ck_types+=("best")
                ck_types+=("best_top$TOPK")
                ;;
            last-best-top-lastk)
                ck_types=("last")
                ck_types+=("best")
                ck_types+=("best_top$TOPK")
                ck_types+=("last$TOPK")
                ;;                
            best)
                ck_types=("best")
                ;;
            last)
                ck_types=("last")
                ;;      
            lastk)
                ck_types=("last$TOPK")
                ;;                                          
            *) 
                echo "checkpoints type id is wrong"
                exit 1    
        esac
      shift 2
      ;;                 
    --) shift; 
        break
  esac
done


echo "========================================================"

if [ "${#exp_array[@]}" -gt 0 ]; then
    echo "List of experiments:"
    for i in "${exp_array[@]}"; do
        # Do what you need based on $i
        echo -e "\t$i"
    done
fi

# while [ : ]; do

#   case "$1" in 
#     -e | --experiment)
#         # experiment_ids="$2"
#         echo "$2"
# 	    read -ra exp_array <<<$2
#         echo "Numbers of experinments : ${#exp_array[@]}"

#         if [ "${#exp_array[@]}" -gt 1 ]; then
#             echo "List of experiments:"
#             for i in "${exp_array[@]}"; do
#                 # Do what you need based on $i
#                 echo -e "\t$i"
#             done
#         fi
#         shift 2
#         ;;
      
#     --) shift; 
#         break
#   esac
# done


# DATA_TYPES=${data_subset[@]}
# CHECK_TYPES=("last" "best" "best_top$TOPK")
ARCH=nat_pretrained_model
CRITERION=nat_ctc_loss
TASK=translation_align_reorder
CHECKPOINTS_PATH=checkpoints

if [ "$load_exist_bleu" = "False" ]; then 
    for i in "${!exp_array[@]}"; do 
        experiment_id=${exp_array[$i]}
        CHECKPOINT=$CHECKPOINTS_PATH/$experiment_id
        if [ ! -d "$CHECKPOINT" ]; then
            # echo "Folder is not exist"
            continue
        fi        
        echo "=========================No.$((i+1))  ID:$experiment_id=============================="
        get_dataset "$experiment_id"
        get_pretrain_model "$experiment_id"
        get_fix_lm_swe "$experiment_id"
        get_voc "$experiment_id"
        get_kd_model "$experiment_id"
        get_ctc "$experiment_id"
        # update_freq=$(((batch_size/max_tokens)/gpu))
        # echo -e "Experiment:$experiment_id \nGPU_Number:$gpu \nBatch_Size:$batch_size \nMax_Tokens:$max_tokens \nMax_Epoch:$max_epoch \nUpdate_Freq:$update_freq"
        # echo -e "Dataset:$dataset  \nPretrained_Model:$pretrained_model \nFix_LM:$fix_lm \nFix_SWE:$fix_swe"
        # echo -e "VOC:$voc \nLM_Loss_Distribution:$lm_loss_dis \nLM_Loss_Layer:$lm_loss_layer \nLM_Loss:$lm_loss"
        # echo -e "Insert_Position:$insert_position \nDY_upsampling:$dynamic_upsampling \nNum_Upsampling_Rate:$num_upsampling_rate \nInsert_Mask:$insert_mask"
        
        BOOL_COMMAND="       "
        # if [ "$fix_lm" = "True" ]
        # then
        #     BOOL_COMMAND+=" --lm-head-frozen"
        # fi
        # if [ "$fix_swe" = "True" ]
        # then
        #     BOOL_COMMAND+=" --embedding-frozen"
        # fi
        # if [ "$lm_loss_dis" = "True" ]
        # then
        #     BOOL_COMMAND+=" --lm-loss-dis"
        # fi
        # if [ "$lm_loss" = "True" ]
        # then
        #     BOOL_COMMAND+=" --lm-loss"
        # fi
        if [ "$dynamic_upsampling" = "True" ]
        then
            BOOL_COMMAND+=" --dynamic-upsampling"
        fi
        if [ "$insert_mask" = "True" ]
        then
            BOOL_COMMAND+=" --upsample-fill-mask"
        fi
        
        if [ "$dynamic_rate" = "True" ]
        then
            BOOL_COMMAND+=" --dynamic-rate"
        fi

        
        if [ "$cpu" = "True" ]
        then
            BOOL_COMMAND+=" --cpu"
        fi
        if [ "$debug" = "True" ]
        then
            BOOL_COMMAND+=" --debug"
        fi    
        if [ "$no_atten_mask" = "True" ]
        then
            BOOL_COMMAND+=" --no-atten-mask"
        fi           

        if [ "$avg_ck_turnoff" = "False" ]; then
            avg_topk_best_checkpoints $CHECKPOINT $TOPK $CHECKPOINT/checkpoint_best_top$TOPK.pt
            avg_lastk_checkpoints $CHECKPOINT $TOPK $CHECKPOINT/checkpoint_last$TOPK.pt
        fi
        

        echo -e "Checkpoint : $CHECKPOINT\t  Batchsize : $batch_size"
    # ---------------------------------------
        for ck_ch in "${ck_types[@]}"; do
            checkpoint_data_name=checkpoint_$ck_ch.pt
            if [ ! -f "$CHECKPOINT/$checkpoint_data_name" ]; then
                echo "$checkpoint_data_name is not exist"
                continue
            fi            
            for data_type in "${data_subset[@]}" ; do         
                for (( speed_i=1; speed_i<=$avg_speed; speed_i++ )); do
                    RESULT_PATH=$CHECKPOINT/${data_type}$no_atten_postfix/${ck_ch}_${batch_size}_${speed_i}.bleu
                    if [ "$skip_exist_genfile" = "True" ]
                    then
                        # Check that the file has been generated.
                        FILE_PATH=$RESULT_PATH/generate-$data_type.txt
                        last_generate_word=$((tail -n1 $FILE_PATH) | awk '{print $1;}')
                        if [ "$last_generate_word" = "Generate" ]
                        then
                            continue
                        fi
                    fi

echo "
CRITERION=$CRITERION
CHECKPOINT=$CHECKPOINTS_PATH/$experiment_id
TASK=$TASK
DATA_BIN=$dataroot/$dataset/de-en-databin
PRETRAINED_MODEL_NAME=$pretrained_model_name
RESULT_PATH=$RESULT_PATH
CHECKPOINTS_DATA=$checkpoint_data_name
DATA_TYPE=$data_type
PRETRAINED_MODE=$pretrained_model
ARCH=$ARCH
BATCH_SIZE=$batch_size
BPE=$bpe

"  > $CHECKPOINT/temp.sh
                
                cat > $CHECKPOINT/temp1.sh << 'endmsg'
        

    python generate.py \
        $DATA_BIN \
        --gen-subset $DATA_TYPE \
        --task $TASK \
        --path $CHECKPOINT/$CHECKPOINTS_DATA \
        --results-path $RESULT_PATH \
        --arch $ARCH \
        --iter-decode-max-iter 0 \
        --criterion $CRITERION \
        --beam 1 \
        --no-repeat-ngram-size 1 \
        --left-pad-source \
        --prepend-bos \
        --pretrained-lm-name $PRETRAINED_MODEL_NAME \
        --pretrained-model-name $PRETRAINED_MODEL_NAME \
        --sacrebleu \
        --bpe $BPE \
        --pretrained-bpe ${PRETRAINED_MODEL_NAME} --pretrained-bpe-src ${PRETRAINED_MODEL_NAME} \
        --remove-bpe \
        --upsample-fill-mask \
        --batch-size $BATCH_SIZE \
endmsg

                cat $CHECKPOINT/temp.sh $CHECKPOINT/temp1.sh > $CHECKPOINT/scrip_generate_$ck_ch.sh
                echo "$BOOL_COMMAND" >> $CHECKPOINT/scrip_generate_$ck_ch.sh
                
                rm $CHECKPOINT/temp*

                bash $CHECKPOINT/scrip_generate_$ck_ch.sh 

                done
            done
        done
    done
fi


#======================Load and Save File==============================
mkdir -p call_scripts/generate/output_file/all_${no_atten_postfix}

csv_file=call_scripts/generate/output_file/output_read.csv
if [ -f "$csv_file" ]; then 
    rm $csv_file
fi

for i in "${!exp_array[@]}"; do 
    experiment_id=${exp_array[$i]}
    CHECKPOINT=$CHECKPOINTS_PATH/$experiment_id
    if [ -d "$CHECKPOINT" ]; then
        echo "=========No.$((i+1))  ID:$experiment_id:============="    
        bleu_array=()
        speed_avg_array=()
        # csv_file=call_scripts/generate/output_file/output_read_${experiment_id}_${no_atten_postfix}.csv
        # python call_scripts/tool/load_checkpoint_step.py $CHECKPOINT 'checkpoint.best_bleu'
        if [ "$local" = "False" ]; then
            checkpoint_bestk_step=$(python call_scripts/tool/load_checkpoint_step.py $CHECKPOINT 'checkpoint.best_bleu')
        fi
        for data_type in "${data_subset[@]}" ; do
            output_bleu_array=()
            output_speed_avg=()
            for ck_ch in "${ck_types[@]}"; do
                checkpoint_data_name=checkpoint_$ck_ch.pt
                if [ ! -f "$CHECKPOINT/$checkpoint_data_name" ]; then
                    echo "$checkpoint_data_name is not exist"
                    continue
                fi  
                # speed_sum=0
                speed_array=()
                for (( speed_i=1; speed_i<=$avg_speed; speed_i++ )); do
                    RESULT_PATH=$CHECKPOINT/${data_type}$no_atten_postfix/${ck_ch}_${batch_size}_${speed_i}.bleu
                    FILE_PATH=$RESULT_PATH/generate-$data_type.txt
                    
                    if [ -f "$FILE_PATH" ]; then
                        lastln=$(tail -n1 $FILE_PATH)
                        last_generate_word=$(tail -n1 $FILE_PATH | awk '{print $1;}')
                        if [ "$last_generate_word" = "Generate" ]
                        then
                            output_bleu=$(echo $lastln | cut -d "=" -f3 | cut -d "," -f1)
                        else
                            output_bleu="Fail"                
                        fi

                        speedln=$(tail -n2 $FILE_PATH | head -1) 
                        last_generate_word=$(tail -n2 $FILE_PATH | head -1 | awk '{print $1;}')
                        if [ "$last_generate_word" = "Translated" ]
                        then
                            output_speed=$(echo $speedln | cut -d " " -f7 | cut -d "s" -f1)                   
                        else
                            output_speed="Fail"                
                        fi                    
                    else 
                        output_bleu="Fail"  
                        output_speed="Fail"               
                    fi
                    speed_array+=("$output_speed ")
                    # sum=$(echo "$speed_sum + $output_speed" | bc)
                done
                # echo "${speed_array[@]}"
                run_avg=$(echo "python call_scripts/tool/avg_speed.py ${speed_array[@]} ")
                avg=$(eval $run_avg | awk '{print $1;}') 
                # avg=$(echo 'scale=5; $sum / $N' | bc -l)
                output_speed_avg+=("$avg/")
                output_bleu_array+=("$output_bleu/")     ## the "/" impact tool/reocrd_score.py
            done
            echo -e "  data-subset: $data_type"
            
            echo -e "\tbleu:\t${output_bleu_array[@]}\t speed:\t${output_speed_avg[@]}\t bestk:\t${checkpoint_bestk_step}" | sed 's/.$//' | sed 's/ //g'     ## the first and second "\t" impact tool/reocrd_score.py
            bleu_array+=$(echo -e "${output_bleu_array[@]}" | sed 's/.$//' | sed 's/ //g'),
            speed_avg_array+=$(echo -e "${output_speed_avg[@]}" | sed 's/.$//' | sed 's/ //g'),
        done
        echo "$experiment_id,bleu,speed_${batch_size},${bleu_array[@]}${speed_avg_array[@]}" >> $csv_file
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