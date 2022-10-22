#!/bin/bash
source $HOME/.bashrc 
conda activate base
#---------Path Setting-------------------#


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
        dataset="iwslt14_ro_en"
    elif [ "$i" = "4" ]
    then
        dataset="iwslt14_en_de"
    else
        echo "error dataset id "
    fi
}

function get_pretrain_model() {
    i=$(echo $1 | cut -d - -f 2)
    if [ "$i" = "1" ]
    then
        pretrained_model="mbert"
        pretrained_model_name="bert-base-multilingual-uncased"
    elif [ "$i" = "2" ]
    then
        pretrained_model="bibert"
        pretrained_model_name="jhu-clsp/bibert-ende"
    elif [ "$i" = "3" ]
    then
        pretrained_model="distill-mbert"
    elif [ "$i" = "4" ]
    then
        pretrained_model="xmlr"
    else
        echo "error pretrained model id "
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

default_setting


VALID_ARGS=$(getopt -o e:g:b: --long experiment:,gpu:,batch_size:,max-tokens:,max-epoch: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1;
fi

eval set -- "$VALID_ARGS"

while [ : ]; do
  case "$1" in 
    -e | --experiment)
      experiment_id="$2"
      shift 2
      ;;
    -g | --gpu)
      gpu="$2"
      shift 2
      ;;   
    -b | --batch_size)
      batch_size="$2"
      shift 2
      ;;    
    --max-tokens)
      max_tokens="$2"
      shift 2
      ;;  
    --max-epoch)
      max_epoch="$2"
      shift 2
      ;;                 
    --) shift; 
        break
  esac
done

get_dataset "$experiment_id"
get_pretrain_model "$experiment_id"
get_fix_lm_swe "$experiment_id"
get_voc "$experiment_id"
get_kd_model "$experiment_id"
get_ctc "$experiment_id"
update_freq=$(((batch_size/max_tokens)/gpu))
# echo "$dataset"
# echo "$pretrained_model"
# echo "$gpu"
# echo "$fix_lm"
# echo "$fix_swe"
# echo "$voc"
# echo "$lm_loss_dis"
# echo "$lm_loss_layer"
# echo "$lm_loss"
# echo "$insert_position"
# echo "$dynamic_upsampling"
# echo "$num_upsampling_rate"
# echo "$insert_mask"
echo -e "Experiment:$experiment_id \nGPU_Number:$gpu \nBatch_Size:$batch_size \nMax_Tokens:$max_tokens \nMax_Epoch:$max_epoch \nUpdate_Freq:$update_freq"
echo -e "Dataset:$dataset  \nPretrained_Model:$pretrained_model \nFix_LM:$fix_lm \nFix_SWE:$fix_swe"
echo -e "VOC:$voc \nLM_Loss_Distribution:$lm_loss_dis \nLM_Loss_Layer:$lm_loss_layer \nLM_Loss:$lm_loss"
echo -e "Insert_Position:$insert_position \nDY_upsampling:$dynamic_upsampling \nNum_Upsampling_Rate:$num_upsampling_rate \nInsert_Mask:$insert_mask"


