source $HOME/.bashrc 
conda activate base

model_root_path=data/pruned_model
model_name=xlmr
SRC=de
TGT=en
model_path=$model_root_path/$model_name
ori_model_path=$model_root_path/$model_name/ori_model
distilled_dataset_path=data/nat_position_reorder/awesome/Bibert_detoken_distill_iwslt14_de_en/demose
raw_data_path=data/nat_position_reorder/awesome/iwslt14_de_en_detoken
mkdir -p $ori_model_path  


# ## download model from huggingface
#=================== XLMR ==================
# wget https://huggingface.co/xlm-roberta-base/resolve/main/pytorch_model.bin -P $ori_model_path
# wget https://huggingface.co/xlm-roberta-base/resolve/main/tokenizer.json -P $ori_model_path
# wget https://huggingface.co/xlm-roberta-base/resolve/main/config.json -P $ori_model_path
# wget https://huggingface.co/xlm-roberta-base/resolve/main/sentencepiece.bpe.model -P $ori_model_path

#=================== mBert =======================================


#=================================================================

gc_file=data/pruned_model/textpurned_config/$model_name/gc.josn
vc_file=data/pruned_model/textpurned_config/$model_name/vc.josn 
 



cat_file=$model_root_path/$model_name/pruned_models
pruned_models_path=$model_root_path/$model_name
mkdir $cat_file 

# #===================cat file=======================#
# cat $distilled_dataset_path/train.en $distilled_dataset_path/train.de \
#     $distilled_dataset_path/valid.en $distilled_dataset_path/valid.de \
#     $raw_data_path/train.en $raw_data_path/train.de \
#     $raw_data_path/valid.en $raw_data_path/valid.de > $cat_file/cat_distill-train-valid_raw-train-valid.txt


# echo "cat demose datat is done"
 
# #=====================xlmr======================#
# textpruner-cli  \
#   --pruning_mode vocabulary \
#   --configurations $gc_file $vc_file \
#   --model_class XLMRobertaForMaskedLM  \
#   --tokenizer_class XLMRobertaTokenizer \
#   --model_path $ori_model_path \
#   --vocabulary $cat_file/cat_distill-train-valid_raw-train-valid.txt

pruned_dir=$(find $cat_file -iname 'pruned_V*' -type d)
python call_scripts/tool/prunmodel/get_voc.py $pruned_dir $pruned_dir 


#=====================mBert==========================#
# textpruner-cli  \
#   --pruning_mode vocabulary \
#   --configurations $gc_file $vc_file \
#   --model_class BertForMaskedLM  \
#   --tokenizer_class BertTokenizer \
#   --model_path $ori_model_path \
#   --vocabulary $cat_file/cat_distill-train-valid_raw-train-valid.txt

 
