source $HOME/.bashrc 
conda activate base
#!/bin/bash
# need prepare
# 1. distill data nd raw data path
# 2. Textprune and gc.json and vc.json

model_name=mBert          #-----input model name {xlmr, mBert}
dataset=iwslt14_deen      #-----input dataset path {iwslt14_deen, wmt14_deen, wmt16_roen, wmt16_enro, wmt16_roen}
download_model=False      #-----if True, download model from huggingface
cat_file_bool=True             #-----if True, cat the file form dataset path

model_root_path=data/pruned_model

cat_file=$model_root_path/$model_name/pruned_models/$dataset
mkdir $cat_file 


pruned_models_path=$model_root_path/$model_name
model_path=$model_root_path/$model_name
ori_model_path=$model_root_path/$model_name/ori_model
gc_file=data/pruned_model/textpurned_config/$model_name/$dataset/gc.json
vc_file=data/pruned_model/textpurned_config/$model_name/$dataset/vc.json 



case $dataset in
    iwslt14_deen)
        # distilled_dataset_path="data/nat_position_reorder/awesome/Bibert_detoken_distill_iwslt14_de_en/demose"
        distilled_dataset_path="data/nat_position_reorder/awesome/baseline_detoken_distill_iwslt14_de_en"
        raw_data_path="data/nat_position_reorder/awesome/iwslt14_en_de_detoken"
        src=de
        tgt=en
        ;;
    wmt14_deen)
        distilled_dataset_path="data/nat_position_reorder/awesome/wmt14_clean_de_en_6kval_BlDist_cased_detoken"
        raw_data_path="data/nat_position_reorder/awesome/wmt14_clean_en_de_detoken" # without wmt14_clean_de_en_detoken
        src=de
        tgt=en     
        ;;
    wmt14_ende)
        distilled_dataset_path="data/nat_position_reorder/awesome/wmt14_clean_en_de_6kval_BlDist_cased_detoken"
        raw_data_path="data/nat_position_reorder/awesome/wmt14_clean_en_de_detoken"
        src=en
        tgt=de        
        ;;        
    wmt16_roen)
        distilled_dataset_path="data/nat_position_reorder/awesome/baseline_detoken_distilled_ro-en/demose"
        raw_data_path="data/nat_position_reorder/awesome/wmt16_en_ro_detoken/demose"
        src=ro
        tgt=en        
        ;;           
    wmt16_enro)
        distilled_dataset_path="data/nat_position_reorder/awesome/baseline_detoken_distilled_en-ro/demose"
        raw_data_path="data/nat_position_reorder/awesome/wmt16_en_ro_detoken/demose"
        src=en
        tgt=ro        
        ;;           
    *)
        echo "Invalid dataset"
        exit 1
        ;;
esac

echo "Distilled dataset path: $distilled_dataset_path"
echo "Raw data path: $raw_data_path"


mkdir -p $ori_model_path  

if [ "$download_model" = "True" ]; then
    case $model_name in
        xlmr)
            # =================== XLMR ==================
            wget https://huggingface.co/xlm-roberta-base/resolve/main/pytorch_model.bin -P $ori_model_path
            wget https://huggingface.co/xlm-roberta-base/resolve/main/tokenizer.json -P $ori_model_path
            wget https://huggingface.co/xlm-roberta-base/resolve/main/config.json -P $ori_model_path
            wget https://huggingface.co/xlm-roberta-base/resolve/main/sentencepiece.bpe.model -P $ori_model_path       
            ;;
        mBert)
            #=====================mBert==========================#

            ;;
        *)
            echo "Invalid model"
            exit 1
            ;;
    esac
    echo "finish download the $model_name file"
fi

if [ "$cat_file_bool" = "True" ]; then
#===================cat file=======================#
    cat $distilled_dataset_path/train.$src $distilled_dataset_path/train.$tgt \
        $distilled_dataset_path/valid.$src $distilled_dataset_path/valid.$tgt \
        $raw_data_path/train.$src $raw_data_path/train.$tgt \
        $raw_data_path/valid.$src $raw_data_path/valid.$tgt > $cat_file/cat_distill-train-valid_raw-train-valid.txt
    
    echo "cat demose datat is done"
fi

case $model_name in
    xlmr)
        #=====================xlmr======================#
        textpruner-cli  \
          --pruning_mode vocabulary \
          --configurations $gc_file $vc_file \
          --model_class XLMRobertaForMaskedLM  \
          --tokenizer_class XLMRobertaTokenizer \
          --model_path $ori_model_path \
          --vocabulary $cat_file/cat_distill-train-valid_raw-train-valid.txt

        pruned_dir=$(find $cat_file -iname 'pruned_V*' -type d)
        python call_scripts/tool/prunmodel/get_voc.py $pruned_dir $pruned_dir           
        ;;
    mBert)
        #=====================mBert==========================#
        textpruner-cli  \
          --pruning_mode vocabulary \
          --configurations $gc_file $vc_file \
          --model_class BertForMaskedLM  \
          --tokenizer_class BertTokenizer \
          --model_path $ori_model_path \
          --vocabulary $cat_file/cat_distill-train-valid_raw-train-valid.txt
        ;;
    *)
        echo "Invalid model"
        exit 1
        ;;
esac



# ## download model from huggingface
#=================== XLMR ==================
# wget https://huggingface.co/xlm-roberta-base/resolve/main/pytorch_model.bin -P $ori_model_path
# wget https://huggingface.co/xlm-roberta-base/resolve/main/tokenizer.json -P $ori_model_path
# wget https://huggingface.co/xlm-roberta-base/resolve/main/config.json -P $ori_model_path
# wget https://huggingface.co/xlm-roberta-base/resolve/main/sentencepiece.bpe.model -P $ori_model_path

#=================== mBert =======================================


#=================================================================

# #=====================xlmr======================#
# textpruner-cli  \
#   --pruning_mode vocabulary \
#   --configurations $gc_file $vc_file \
#   --model_class XLMRobertaForMaskedLM  \
#   --tokenizer_class XLMRobertaTokenizer \
#   --model_path $ori_model_path \
#   --vocabulary $cat_file/cat_distill-train-valid_raw-train-valid.txt

# pruned_dir=$(find $cat_file -iname 'pruned_V*' -type d)
# python call_scripts/tool/prunmodel/get_voc.py $pruned_dir $pruned_dir 


#=====================mBert==========================#
# textpruner-cli  \
#   --pruning_mode vocabulary \
#   --configurations $gc_file $vc_file \
#   --model_class BertForMaskedLM  \
#   --tokenizer_class BertTokenizer \
#   --model_path $ori_model_path \
#   --vocabulary $cat_file/cat_distill-train-valid_raw-train-valid.txt

 
