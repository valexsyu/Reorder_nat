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
        dataset="distill_iwslt14_de_en_mbert"
    elif [ "$i" = "2" ]
    then
        dataset="distill_iwslt14_de_en_bibert"
    elif [ "$i" = "3" ]
    then
        dataset="distill_baseline_iwslt14_de_en_mbert"
    elif [ "$i" = "4" ]
    then
        dataset="distill_baseline_iwslt14_de_en_bibert"
    elif [ "$i" = "5" ]
    then
        dataset="distill_iwslt14_de_en_dmbert"
    elif [ "$i" = "6" ]
    then
        dataset="distill_baseline_iwslt14_de_en_dmbert"
    elif [ "$i" = "7" ]
    then
        dataset="distill_iwslt14_de_en_xlmr"
    elif [ "$i" = "8" ]
    then
        dataset="distill_baseline_iwslt14_de_en_xlmr"                                
    elif [ "$i" = "A" ]
    then
        dataset="distill_baseline_wmt16_en_ro_mbert"                                
    elif [ "$i" = "B" ]
    then
        dataset="distill_baseline_wmt16_ro_en_mbert"                                
    elif [ "$i" = "C" ]
    then
        dataset="distill_fnc_wmt16_ro_en_mbert"           
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
        BATCH_SIZE=60
    elif [ "$i" = "2" ]
    then
        pretrained_model="bibert"
        pretrained_model_name="jhu-clsp/bibert-ende"
        bpe="bibert"
        BATCH_SIZE=60
    elif [ "$i" = "3" ]
    then
        pretrained_model="dmbert"
        pretrained_model_name="distilbert-base-multilingual-cased"
        bpe="bibert"
        BATCH_SIZE=60
    elif [ "$i" = "4" ]
    then
        pretrained_model="xlmr"
        pretrained_model_name="xlm-roberta-base"
        bpe="xlmr"
        BATCH_SIZE=20
    elif [ "$i" = "5" ]
    then
        pretrained_model="mbert"
        pretrained_model_name="bert-base-multilingual-uncased"     
        bpe="bibert"
        BATCH_SIZE=60      
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
    batch_size=12288
    max_tokens=2048
    max_epoch=400
    update_freq=6
    
}

function avg_topk_best_checkpoints(){
	python scripts/average_checkpoints.py \
	            --inputs $1 \
				--num-epoch-checkpoints $2 --output $3 \
				--ckpref checkpoints.best_bleu	
}


default_setting

VALID_ARGS=$(getopt -o e: --long experiment: -- "$*")
if [[ $? -ne 0 ]]; then
    exit 1;
fi

eval set -- "$VALID_ARGS"
while [ : ]; do

  case "$1" in 
    -e | --experiment)
        # experiment_ids="$2"
        echo "$2"
	    read -ra exp_array <<<$2
        echo "Numbers of experinments : ${#exp_array[@]}"

        if [ "${#exp_array[@]}" -gt 1 ]; then
            echo "List of experiments:"
            for i in "${exp_array[@]}"; do
                # Do what you need based on $i
                echo -e "\t$i"
            done
        fi
        shift 2
        ;;
      
    --) shift; 
        break
  esac
done


TOPK=5

DATA_TYPES=("test")
CHECK_TYPES=("last" "best" "best_top$TOPK")
ARCH=nat_pretrained_model
CRITERION=nat_ctc_loss
TASK=translation_align_reorder
CHECKPOINTS_PATH=checkpoints

for i in "${!exp_array[@]}"; do 
    experiment_id=${exp_array[$i]}
    echo "=========================No.$((i+1))  ID:$experiment_id=============================="
    get_dataset "$experiment_id"
    get_pretrain_model "$experiment_id"
    get_fix_lm_swe "$experiment_id"
    get_voc "$experiment_id"
    get_kd_model "$experiment_id"
    get_ctc "$experiment_id"
    update_freq=$(((batch_size/max_tokens)/gpu))
    # echo -e "Experiment:$experiment_id \nGPU_Number:$gpu \nBatch_Size:$batch_size \nMax_Tokens:$max_tokens \nMax_Epoch:$max_epoch \nUpdate_Freq:$update_freq"
    # echo -e "Dataset:$dataset  \nPretrained_Model:$pretrained_model \nFix_LM:$fix_lm \nFix_SWE:$fix_swe"
    # echo -e "VOC:$voc \nLM_Loss_Distribution:$lm_loss_dis \nLM_Loss_Layer:$lm_loss_layer \nLM_Loss:$lm_loss"
    # echo -e "Insert_Position:$insert_position \nDY_upsampling:$dynamic_upsampling \nNum_Upsampling_Rate:$num_upsampling_rate \nInsert_Mask:$insert_mask"
     
    BOOL_COMMAND="        "
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





    CHECKPOINT=$CHECKPOINTS_PATH/$experiment_id


    avg_topk_best_checkpoints $CHECKPOINT $TOPK $CHECKPOINT/checkpoint_best_top$TOPK.pt
    

    echo -e "Checkpoint : $CHECKPOINT\t  Batchsize : $BATCH_SIZE"
# ---------------------------------------
    for ck_ch in "${CHECK_TYPES[@]}"; do
        for data_type in "${DATA_TYPES[@]}" ; do
        echo "
        CRITERION=$CRITERION
        CHECKPOINT=$CHECKPOINTS_PATH/$experiment_id
        TASK=$TASK
        DATA_BIN=/livingrooms/valexsyu/dataset/nat/$dataset/de-en-databin
        PRETRAINED_MODEL_NAME=$pretrained_model_name
        RESULT_PATH=$CHECKPOINT/$data_type/$ck_ch.bleu/
        CHECKPOINTS_DATA=checkpoint_$ck_ch.pt
        DATA_TYPE=$data_type
        PRETRAINED_MODE=$pretrained_model
        ARCH=$ARCH
        BATCH_SIZE=$BATCH_SIZE
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

        cat $CHECKPOINT/temp.sh $CHECKPOINT/temp1.sh > $CHECKPOINT/scrip_generate_$CHECK_TYPES.sh
        echo "$BOOL_COMMAND" >> $CHECKPOINT/scrip_generate_$CHECK_TYPES.sh

        rm $CHECKPOINT/temp*

        bash $CHECKPOINT/scrip_generate_$CHECK_TYPES.sh          
        done
    done
done








