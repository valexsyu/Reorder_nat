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
    elif [ "$i" = "i" ]
    then
        dataset="iwslt14_de_en_BlDist_cased_mbert"         
    elif [ "$i" = "j" ]
    then
        dataset="wmt14_en_de_BigBlDist_xlmr"       
    elif [ "$i" = "k" ]
    then
        dataset="iwslt14_de_en_BlDist_mbertcased"            
    elif [ "$i" = "m" ]
    then
        dataset="iwslt14_de_en_bibertDist_mbert_pruned26458"            
    elif [ "$i" = "n" ]
    then
        dataset="iwslt14_de_en_bibertDist_mbert_pruned26458_8k"
    elif [ "$i" = "o" ]
    then
        dataset="iwslt14_de_en_bibertDist_xlmr_pruned21785"     
    elif [ "$i" = "p" ]
    then
        dataset="iwslt14_de_en_mbert_pruned26458"            
    else      
        echo "error dataset id "
        echo $1
        exit 1
    fi
} 



function get_pretrain_model() {
    i=$(echo $1 | cut -d - -f 2)
    if [ "$i" = "1" ]
    then
        pretrained_model="mbert"
        pretrained_model_name="bert-base-multilingual-uncased"
        init_translator=False
    elif [ "$i" = "2" ]
    then
        pretrained_model="bibert"
        pretrained_model_name="jhu-clsp/bibert-ende"
        init_translator=False
    elif [ "$i" = "3" ]
    then
        pretrained_model="dmbert"
        pretrained_model_name="distilbert-base-multilingual-cased"
        init_translator=False
    elif [ "$i" = "4" ]
    then
        pretrained_model="xlmr"
        pretrained_model_name="xlm-roberta-base"
        init_translator=False
    elif [ "$i" = "5" ]
    then
        pretrained_model="mbert"
        pretrained_model_name="bert-base-multilingual-uncased"     
        init_translator=True
    elif [ "$i" = "6" ]
    then
        pretrained_model="bibert"
        pretrained_model_name="jhu-clsp/bibert-ende"
        init_translator=True     
    elif [ "$i" = "7" ]
    then
        pretrained_model="mbert-cased"
        pretrained_model_name="bert-base-multilingual-cased"
        init_translator=False      
    elif [ "$i" = "8" ]
    then
        pretrained_model="mbert"
        pretrained_model_name="bert-base-multilingual-uncased"
        init_translator=False   
        pretrained_lm_path=$modelroot/mbert/pruned_models_BertModel/pruned_V26458/
        pretrained_model_path=$modelroot/mbert/pruned_models_BertForMaskedLM/pruned_V26458  
    elif [ "$i" = "A" ]
    then
        pretrained_model="mbert"
        pretrained_model_name="bert-base-multilingual-uncased"
        bpe="bibert"    
        pretrained_lm_path=$modelroot/mbert/pruned_models_BertModel/pruned_V26458/
        pretrained_model_path=$modelroot/mbert/pruned_models_BertModel/pruned_V26458/      
    elif [ "$i" = "B" ]
    then
        pretrained_model="mbert"
        pretrained_model_name="bert-base-multilingual-uncased"
        bpe="bibert"    
        pretrained_lm_path=$modelroot/mbert/pruned_models_BertForMaskedLM/pruned_V26458/ 
        pretrained_model_path=$modelroot/mbert/pruned_models_BertForMaskedLM/pruned_V26458/           
    elif [ "$i" = "C" ]
    then
        pretrained_model="xlmr"
        pretrained_model_name="xlm-roberta-base"
        bpe="xlmr"    
        pretrained_lm_path=$modelroot/xlmr/pruned_models_BertForMaskedLM/pruned_21785/ 
        pretrained_model_path=$modelroot/xlmr/pruned_models_BertForMaskedLM/pruned_21785/                        
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
    elif [ "$i" = "4" ]
    then
        voc="4"      
    elif [ "$i" = "5" ]
    then
        voc="5"           
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
        lm_loss_type=COS
        lm_loss=True     
        lmk_loss=False   
    else
        if [ "$i" = "T" ]
        then
            lm_loss_type=DIS
            lm_loss_layer=-1
            lm_loss=True
            lmk_loss=False 
        elif [ "$i" = "N" ]
        then
            lm_loss_type=COS
            lm_loss_layer=-1
            lm_loss=False
            lmk_loss=False 
        elif [ $(echo $i | cut -c 1) = "K" ]
        then
            lm_loss_type=COS
            lm_loss_layer=$(($(echo $i | cut -c 2-3)-13))
            lm_loss=True     
            lmk_loss=True     
        elif [ $(echo $i | cut -c 1) = "A" ]
        then
            lm_loss_layer=$(($(echo $i | cut -c 2-3)-13))
            lm_loss_type=COS
            lm_loss=True     
            lmk_loss=False 
            lm_random_mask=True   
        elif [ $(echo $i | cut -c 1) = "B" ]
        then
            lm_loss_layer=$(($(echo $i | cut -c 2-3)-13))
            lm_loss_type=COS
            lm_loss=True     
            lmk_loss=True 
            lm_random_mask=True       
        elif [ $(echo $i | cut -c 1) = "C" ]
        then
            lm_loss_layer=$(($(echo $i | cut -c 2-3)-13))
            lm_loss_type=DIS
            lm_loss=True     
            lmk_loss=False 
            lm_random_mask=True                                            
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
        B)
            insert_mask=True
            blank_use_mask=True
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
    dryrun=False
    train_subset=train
    max_update=100000
    dataroot=/livingrooms/valexsyu/dataset/nat  
    modelroot=/livingrooms/valexsyu/dataset/model
    pretrained_model_path=None
    pretrained_lm_path=None
    fp16=False  
    save_interval_updates=10000
    dropout=0.1
    lm_start_step=75000
    no_atten_mask=False
    twcc=False
    watch_test_bleu=False
    warmup_updates=10000
    reset_dataloader=False
    reset_optimizer=False
    reset_lr_scheduler=False
    reset_meters=False
    debug=False
    has_eos=False
    blank_use_mask=False
    lm_random_mask=False
    wandb_team_id=valex-jcx
    lm_iter_num=1
    lm_loss_type=COS
    lm_mask_rate=-1
    arch=nat_pretrained_model
    criterion=nat_ctc_loss
    task=translation_ctcpmlm
    lmax_only_step=5000
    hydra=False
    
}

default_setting

VALID_ARGS=$(getopt \
            -o e:g:b:s: \
            --long experiment:,gpu:,batch-size:,dryrun,max-tokens:,max-epoch:,max-update:,twcc,local,fp16,valid-set,\
            --long save-interval-updates:,dropout:,lm-start-step:,no-atten-mask,watch-test-bleu,warmup-updates:, \
            --long reset-dataloader,reset-optimizer,debug,has-eos,wandb-team-id:,lm-iter-num:,watch-lm-loss,lm-mask-rate:, \
            --long reset-meters,reset-lr-scheduler,arch:,criterion:,lmax-only-step:,task:,hydra \
             -- "$@")
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
    -b | --batch-size)
      batch_size="$2"
      shift 2
      ;;    
    --dryrun)
      dryrun=True
      shift 1
      ;;             
    --max-tokens)
      max_tokens="$2"
      shift 2
      ;;  
    --max-epoch)
      max_epoch="$2"
      shift 2
      ;;  
    --max-update)
      max_update="$2"
      shift 2
      ;;           
    -s | --save-interval-updates)
      save_interval_updates="$2"
      shift 2
      ;;          
    --twcc)
      dataroot="../nat_data"
      twcc=True
      shift 1
      ;;      
    --local)
      dataroot="../../dataset/nat"
      modelroot="../../dataset/model"      
      shift 1
      ;;         
    --watch-test-bleu)
      watch_test_bleu=True
      shift 1
      ;;        
    --fp16)
      fp16=True
      shift 1
      ;;        
    --valid-set)      
      train_subset=valid
      dryrun=True
      shift 1
      ;;       
    --hydra)      
      hydra=True
      shift 1
      ;;           
    --dropout)
      dropout="$2"
      shift 2
      ;;     
    --lm-start-step)
      lm_start_step="$2"
      shift 2
      ;;  
    --warmup-updates)
      warmup_updates="$2"
      shift 2
      ;;        
    --wandb-team-id)
      wandb_team_id="$2"
      shift 2
      ;;    
    --lm-iter-num)
      lm_iter_num="$2"
      shift 2
      ;;  
    --lm-mask-rate)
      lm_mask_rate="$2"
      shift 2
      ;;   
    --arch)
      arch="$2"
      shift 2
      ;;        
    --task)
      task="$2"
      shift 2
      ;;
    --criterion)
      criterion="$2"
      shift 2
      ;;         
    --lmax-only-step)
      lmax_only_step="$2"
      shift 2
      ;;                                              
    --no-atten-mask)
      no_atten_mask=True
      shift 1
      ;;          
    --reset-dataloader)
      reset_dataloader=True
      shift 1
      ;;   
    --reset-optimizer)
      reset_optimizer=True
      shift 1
      ;;   
    --reset-meters)
      reset_meters=True
      shift 1
      ;;   
    --reset-lr-scheduler)
      reset_lr_scheduler=True
      shift 1
      ;;               
    --debug)
      debug=True   
      shift 1
      ;;          
    --has-eos)
      has_eos=True
      shift 1
      ;;    
    --watch-lm-loss)
      watch_lm_loss=True
      shift 1
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
echo -e "VOC:$voc \nLM_Loss_Type:$lm_loss_type \nLM_Loss_Layer:$lm_loss_layer \nLM_Loss:$lm_loss \nLM_K_Loss:$lmk_loss"
echo -e "Insert_Position:$insert_position \nDY_upsampling:$dynamic_upsampling \nNum_Upsampling_Rate:$num_upsampling_rate \nInsert_Mask:$insert_mask"
echo -e "Init_Translator:$init_translator "

BOOL_COMMAND="   "
if [ "$fix_lm" = "True" ]
then
    BOOL_COMMAND+=" --lm-head-frozen"
fi
if [ "$fix_swe" = "True" ]
then
    BOOL_COMMAND+=" --embedding-frozen"
fi

if [ "$lm_loss" = "True" ]
then
    BOOL_COMMAND+=" --lm-loss"
fi

if [ "$lmk_loss" = "True" ]
then
    BOOL_COMMAND+=" --lmk-loss"
fi

if [ "$dynamic_upsampling" = "True" ]
then
    BOOL_COMMAND+=" --dynamic-upsampling"
fi
if [ "$insert_mask" = "True" ]
then
    BOOL_COMMAND+=" --upsample-fill-mask"
fi

if [ "$init_translator" = "True" ]
then
    BOOL_COMMAND+=" --init-translator"
fi

if [ "$dynamic_rate" = "True" ]
then
    BOOL_COMMAND+=" --dynamic-rate"
fi

if [ "$dryrun" = "False" ]
then
    BOOL_COMMAND+="  --wandb-project"
    BOOL_COMMAND+=" NAT-Pretrained-Model"
    BOOL_COMMAND+="  --wandb-entity"
    BOOL_COMMAND+=" $wandb_team_id"
fi

if [ "$fp16" = "True" ]
then
    BOOL_COMMAND+=" --fp16"
fi
if [ "$no_atten_mask" = "True" ]
then
    BOOL_COMMAND+=" --no-atten-mask"
fi   
if [ "$twcc" = "True" ]
then
    BOOL_COMMAND+=" --twcc"
fi  
if [ "$watch_test_bleu" = "True" ]
then
    BOOL_COMMAND+=" --watch-test-bleu"
fi  
if [ "$reset_dataloader" = "True" ]
then
    BOOL_COMMAND+=" --reset-dataloader"
fi  
if [ "$reset_optimizer" = "True" ]
then
    BOOL_COMMAND+=" --reset-optimizer"
fi  
if [ "$reset_lr_scheduler" = "True" ]
then
    BOOL_COMMAND+=" --reset-lr-scheduler"
fi 
if [ "$reset_meters" = "True" ]
then
    BOOL_COMMAND+=" --reset-meters"
fi   
if [ "$debug" = "True" ]
then
    BOOL_COMMAND+=" --debug"
fi  
if [ "$has_eos" = "True" ]
then
    BOOL_COMMAND+=" --has-eos"
fi 

if [ "$blank_use_mask" = "True" ]
then
    BOOL_COMMAND+=" --blank-use-mask"
fi 

if [ "$lm_random_mask" = "True" ]
then
    BOOL_COMMAND+=" --lm-random-mask"
fi 


if [ "$watch_lm_loss" = "True" ]
then
    BOOL_COMMAND+=" --watch-lm-loss"
fi 

if [ $(echo "$lm_mask_rate != -1"|bc ) -eq 1 ]
then
    BOOL_COMMAND+="  --lm-mask-rate  $lm_mask_rate"
fi 



if [ ! -d "checkpoints" ]; then
    mkdir checkpoints
fi

CHECKPOINT=checkpoints/$experiment_id

# if [[ "$hydra" = "True" && "$dataroot" = "../../dataset/nat" ]]; then
#     DATA_BIN=/home/valex/Documents/Study/dataset/nat/$dataset/de-en-databin
# else
#     DATA_BIN=$dataroot/$dataset/de-en-databin
# fi
DATA_BIN=$dataroot/$dataset/de-en-databin


##----------RUN  Bash-----------------------------
mkdir $CHECKPOINT
mkdir $CHECKPOINT/tensorboard
echo "
GPU=$gpu
EXPERIMENT_ID=$experiment_id
CHECKPOINT=$CHECKPOINT
DATA_BIN=$DATA_BIN
MAX_TOKENS=$max_tokens
MAX_EPOCH=$max_epoch
PRETRAINED_MODEL_NAME=$pretrained_model_name
PRETRAINED_LM_NAME=$pretrained_model_name
PRETRAINED_LM_PATH=$pretrained_lm_path
PRETRAINED_MODEL_PATH=$pretrained_model_path
UPDATE_FREQ=$update_freq
FIX_LM=$fix_lm
FIX_SWE=$fix_swe
LM_LOSS=$lm_loss
LM_K_LOSS=$lmk_loss
LM_LOSS_LAYER=$lm_loss_layer
INSERT_MASK=$insert_mask
NUM_UPSAMPLING_RATE=$num_upsampling_rate
INSERT_POSITION=$insert_position
TRAIN_SUBSET=$train_subset
MAX_UPDATE=$max_update
SAVE_INTERVAL_UPDATES=$save_interval_updates
DROPOUT=$dropout
LM_START_STEP=$lm_start_step
WARMUP_UPDATES=$warmup_updates
VOC_CHOOSEN=$voc
LM_ITER_NUM=$lm_iter_num
LM_LOSS_TYPE=$lm_loss_type
ARCH=$arch
CRITERION=$criterion
LMAX_ONLY_STEP=$lmax_only_step
TASK=$task


"  > $CHECKPOINT/temp.sh
cp $CHECKPOINT/temp.sh $CHECKPOINT/temp_hydra.sh



echo "python train.py \\" >> $CHECKPOINT/temp.sh


cat > $CHECKPOINT/temp1.sh << 'endmsg'
    $DATA_BIN \
    --save-dir $CHECKPOINT \
    --ddp-backend=legacy_ddp \
    --task $TASK \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0002 --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-09' --warmup-updates $WARMUP_UPDATES \
    --warmup-init-lr '1e-07' \
    --weight-decay 0.01 \
    --dropout $DROPOUT \
    --log-format 'simple' --log-interval 100 \
    --fixed-validation-seed 7 \
    --save-interval-updates $SAVE_INTERVAL_UPDATES \
	--criterion $CRITERION \
	--arch $ARCH \
    --tensorboard-logdir $CHECKPOINT/tensorboard \
    --noise no_noise \
    --no-epoch-checkpoints \
    --save-interval 1 \
    --left-pad-source \
    --prepend-bos \
    --align-position-pad-index 513 \
    --keep-best-checkpoints 5 \
    --eval-bleu-print-samples \
    --eval-bleu --eval-bleu-remove-bpe \
    --max-update $MAX_UPDATE \
    --lm-start-step $LM_START_STEP \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --pretrained-lm-name $PRETRAINED_LM_NAME \
    --pretrained-model-name $PRETRAINED_MODEL_NAME \
    --max-tokens $MAX_TOKENS \
    --update-freq $UPDATE_FREQ \
    --num-upsampling-rate $NUM_UPSAMPLING_RATE \
    --insert-position $INSERT_POSITION \
    --train-subset $TRAIN_SUBSET \
    --voc-choosen $VOC_CHOOSEN \
    --lm-iter-num $LM_ITER_NUM \
    --lm-loss-type $LM_LOSS_TYPE \
    --keep-last-epochs 5 \
    --lmax-only-step $LMAX_ONLY_STEP \
    --pretrained-lm-path $PRETRAINED_LM_PATH --pretrained-model-path $PRETRAINED_MODEL_PATH \
endmsg

cat $CHECKPOINT/temp.sh $CHECKPOINT/temp1.sh > $CHECKPOINT/scrip.sh
echo "$BOOL_COMMAND" >> $CHECKPOINT/scrip.sh



#=====================hydra-train=============
echo "python  call_scripts/tool/config_yaml.py \\" >> $CHECKPOINT/temp_hydra.sh

cat > $CHECKPOINT/temp_hydra1.sh << 'endmsg'
    -e $EXPERIMENT_ID \
    --basic-yaml-path "call_scripts/train/basic.yaml" \
    --save-dir $CHECKPOINT \
    --save-interval-updates $SAVE_INTERVAL_UPDATES \
    --task $TASK \
    --data $DATA_BIN \
	--criterion $CRITERION \
	--model-name $ARCH \
    --max-tokens $MAX_TOKENS \
    --max-update $MAX_UPDATE \
    --update-freq $UPDATE_FREQ \
    --num-upsampling-rate $NUM_UPSAMPLING_RATE \
    --train-subset $TRAIN_SUBSET \
    --lm-start-step $LM_START_STEP \
    --voc-choosen $VOC_CHOOSEN \
    --lmax-only-step $LMAX_ONLY_STEP \
    --dropout $DROPOUT \
    --distributed-world-size $GPU \
    --pretrained-lm-path $PRETRAINED_LM_PATH --pretrained-model-path $PRETRAINED_MODEL_PATH \
endmsg

cat $CHECKPOINT/temp_hydra.sh $CHECKPOINT/temp_hydra1.sh > $CHECKPOINT/hydra.sh
echo "$BOOL_COMMAND" >> $CHECKPOINT/hydra.sh
rm $CHECKPOINT/temp*

if [ "$hydra" = "True" ]
then
    bash $CHECKPOINT/hydra.sh
    fairseq-hydra-train -r --config-dir $CHECKPOINT/  --config-name $experiment_id.yaml
else
    bash $CHECKPOINT/scrip.sh
fi 



# :<<'END_COMMENT' 
    # --lm-head-frozen $FIX_LM \
    # --embedding-frozen $FIX_SWE \
    # --lm-loss $LM_LOSS \
    # --lm-loss-layer $LM_LOSS_LAYER \
    # --lm-loss-dis $LM_LOSS_DIS \
    # --upsample-fill-mask $INSERT_MASK \
    # --max-epoch $MAX_EPOCH \
    # --save-interval 1 \
    #----- store last 5 epoch-----------
    # --keep-last-epochs 5 \ add it 
    # --no-epoch-checkpoints \ remove it
    #-----------------------------


    