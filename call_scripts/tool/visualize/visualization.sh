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
    elif [ "$i" = "z" ]
    then
        dataset="iwslt14_de_en_bibertDist_mbert_pruned26458-test"                                                                          
    else        
        echo "error dataset id "
        exit 1
    fi
}


function default_setting() {
    dataroot="../../dataset/nat"
    
    RESULT_PATH=call_scripts/tool/visualize/visualization/
    ARCH=nat_pretrained_model
    CRITERION=nat_ctc_loss
    TASK=translation_align_reorder    
    CHECKPOINTS_PATH=checkpoints
    # CK_NAME=checkpoint_best_top5
    CK_NAME=checkpoint_last
    PRETRAINED_MODEL_NAME="bert-base-multilingual-uncased"
    BATCH_SIZE=50 
    modelroot="../../dataset/model"   
    PRETRAINED_LM_PATH=$modelroot/mbert/pruned_models_BertModel/pruned_V26458/
    PRETRAINED_MODEL_PATH=$modelroot/mbert/pruned_models_BertForMaskedLM/pruned_V26458    
    BPE="bibert" 
    CRITERION=nat_ctc_loss
}

default_setting
VALID_ARGS=$(getopt -o e: --long experiment:,twcc,sleep: -- "$@")
if [[ $? -ne 0 ]]; then
    echo "Error"
    exit 1;
fi

eval set -- "$VALID_ARGS"
default_setting

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
    --sleep)
      sleep_time=$2
      shift 2
      ;;                
    --) shift; 
        break
  esac
done


for i in "${!exp_array[@]}"; do 
    experiment_id=${exp_array[$i]}
    CHECKPOINT=$CHECKPOINTS_PATH/$experiment_id
    
    echo "$CHECKPOINT"
    if [ ! -d "$CHECKPOINT" ]; then
        # echo "Folder is not exist"
        continue
    fi        
    echo "=========================No.$((i+1))  ID:$experiment_id=============================="
    get_dataset "$experiment_id"
done    
DATA_BIN=$dataroot/$dataset/de-en-databin
CUDA_VISIBLE_DEVICES=0 python call_scripts/tool/visualize/visualize_representation.py  $DATA_BIN \
                                    --results-path $RESULT_PATH \
                                    --path $CHECKPOINT/$CK_NAME.pt \
                                    --arch $ARCH \
                                    --task $TASK \
                                    --pretrained-lm-name $PRETRAINED_MODEL_NAME \
                                    --pretrained-model-name $PRETRAINED_MODEL_NAME \
                                    --batch-size $BATCH_SIZE \
                                    --pretrained-lm-path $PRETRAINED_LM_PATH \
                                    --pretrained-model-path $PRETRAINED_MODEL_PATH \
                                    --upsample-fill-mask \
                                    --sacrebleu \
                                    --bpe $BPE \
                                    --iter-decode-max-iter 0 \
                                    --criterion $CRITERION \
                                    --beam 1 \
                                    --no-repeat-ngram-size 1 \
                                    --left-pad-source \
                                    --prepend-bos                                                                       