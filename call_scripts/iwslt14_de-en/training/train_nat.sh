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
    echo "$gpu"
    
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
echo -e "Experiment:$experiment_id \nGPU_Number:$gpu \nBatch_Size:$batch_size \nMax_Tokens:$max_tokens \nMax_Epoch:$max_epoch \nUpdate_Freq:$update_freq"
echo -e "Dataset:$dataset  \nPretrained_Model:$pretrained_model \nFix_LM:$fix_lm \nFix_SWE:$fix_swe"
echo -e "VOC:$voc \nLM_Loss_Distribution:$lm_loss_dis \nLM_Loss_Layer:$lm_loss_layer \nLM_Loss:$lm_loss"
echo -e "Insert_Position:$insert_position \nDY_upsampling:$dynamic_upsampling \nNum_Upsampling_Rate:$num_upsampling_rate \nInsert_Mask:$insert_mask"

BOOL_COMMAND="   "
if [ "$fix_lm" = "True" ]
then
    BOOL_COMMAND+=" --lm-head-frozen"
fi
if [ "$fix_swe" = "True" ]
then
    BOOL_COMMAND+=" --embedding-frozen"
fi
if [ "$lm_loss_dis" = "True" ]
then
    BOOL_COMMAND+=" --lm-loss-dis"
fi
if [ "$lm_loss" = "True" ]
then
    BOOL_COMMAND+=" --lm-loss"
fi
if [ "$dynamic_upsampling" = "True" ]
then
    BOOL_COMMAND+=" --dynamic-upsampling"
fi
if [ "$insert_mask" = "True" ]
then
    BOOL_COMMAND+=" --upsample-fill-mask"
fi


CHECKPOINT=checkpoints/$experiment_id
DATA_BIN=/livingrooms/valexsyu/dataset/nat/$dataset/de-en-databin


##----------RUN  Bash-----------------------------
mkdir $CHECKPOINT
mkdir $CHECKPOINT/tensorboard
echo "
CHECKPOINT=$CHECKPOINT
DATA_BIN=$DATA_BIN
MAX_TOKENS=$max_tokens
MAX_EPOCH=$max_epoch
PRETRAINED_MODEL_NAME=$pretrained_model_name
PRETRAINED_LM_NAME=$pretrained_model_name
UPDATE_FREQ=$update_freq
FIX_LM=$fix_lm
FIX_SWE=$fix_swe
LM_LOSS=$lm_loss
LM_LOSS_LAYER=$lm_loss_layer
LM_LOSS_DIS=$lm_loss_dis
INSERT_MASK=$insert_mask
NUM_UPSAMPLING_RATE=$num_upsampling_rate


"  > $CHECKPOINT/temp.sh

echo "python train.py \\" >> $CHECKPOINT/temp.sh

cat > $CHECKPOINT/temp1.sh << 'endmsg'
    $DATA_BIN \
    --save-dir $CHECKPOINT \
    --ddp-backend=legacy_ddp \
    --task translation_align_reorder \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0002 --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' \
    --weight-decay 0.01 \
    --log-format 'simple' --log-interval 100 \
    --fixed-validation-seed 7 \
    --save-interval-updates 10000 \
	--criterion nat_ctc_loss \
	--arch nat_pretrained_model \
    --tensorboard-logdir $CHECKPOINT/tensorboard \
    --no-epoch-checkpoints \
    --noise no_noise \
    --save-interval 1 \
    --left-pad-source \
    --prepend-bos \
    --align-position-pad-index 513 \
    --keep-best-checkpoints 5 \
    --eval-bleu-print-samples \
    --eval-bleu --eval-bleu-remove-bpe \
    --wandb-project NAT-Pretrained-Model \
    --wandb-entity valex-jcx \
    --max-update 100000 \
    --lm-start-step 75000 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --pretrained-lm-name $PRETRAINED_LM_NAME \
    --pretrained-model-name $PRETRAINED_MODEL_NAME \
    --max-tokens $MAX_TOKENS \
    --max-epoch $MAX_EPOCH \
    --update-freq $UPDATE_FREQ \
    --num-upsampling-rate $NUM_UPSAMPLING_RATE \
    --train-subset train \
endmsg

cat $CHECKPOINT/temp.sh $CHECKPOINT/temp1.sh > $CHECKPOINT/scrip.sh
echo "$BOOL_COMMAND" >> $CHECKPOINT/scrip.sh

rm $CHECKPOINT/temp*

bash $CHECKPOINT/scrip.sh

# :<<'END_COMMENT' 
    # --lm-head-frozen $FIX_LM \
    # --embedding-frozen $FIX_SWE \
    # --lm-loss $LM_LOSS \
    # --lm-loss-layer $LM_LOSS_LAYER \
    # --lm-loss-dis $LM_LOSS_DIS \
    # --upsample-fill-mask $INSERT_MASK \
