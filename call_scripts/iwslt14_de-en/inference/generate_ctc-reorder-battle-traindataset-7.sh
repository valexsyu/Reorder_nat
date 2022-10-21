source $HOME/.bashrc 
conda activate base
# --bpe bibert \
# --pretrained-bpe ${PRE} --pretrained-bpe-src ${PRE_SRC} \
# --pretrained-model-name bert-base-multilingual-uncased \
# --use-align-position \
# -----------    Setting    -------------
# CHECKPOINTS=("No-7-1-000-translation-init" "No-7-1-003-translation-init-lm" "No-7-1-00-translation" "No-7-1-03-translation")
CHECKPOINTS=( "No-7-2-34-translation_mask" "No-7-2-35-translation-lm_mask" "No-7-4-34-translation_mask")
REORDER_TRANSLATION=translation
# REORDER_TRANSLATION=reorder_translation
# DATA_TYPES=("test" "valid" "train")
TOPK=5
DATA_TYPES=("test")
# CHECK_TYPES=("last" "best" "best_top$TOPK")
CHECK_TYPES=("best_top$TOPK")
DATA=iwslt14.de-en
#DATABIN=data/nat_position_reorder/awesome/Bibert_token_distill_iwslt14_de_en/de-en-databin #Bin Data of TEST dataset
# DATABIN=data/nat_position_reorder/awesome/Bibert_token_distill_iwslt14_de_en_52k/de-en-databin #Bin Data of TEST dataset
# DATABIN=data/nat_position_reorder/awesome/Bibert_token_distill_iwslt14_de_en_mbert/de-en-databin #Bin Data of TEST dataset
# DATABIN=data/nat_position_reorder/awesome/Bibert_token_distill_iwslt14_de_en_52k/de-en-databin
DATABIN=data/nat_position_reorder/awesome/Bibert_token_distill_valid-nondistill_iwslt14_de_en_mbert/de-en-databin
CHECKPOINTS_PATH=checkpoints
BATCH_SIZE=62

ARCH=nat_pretrained_model
#ARCH=nat_position_reorder_samll
CRITERION=nat_ctc_loss
TASK=translation_align_reorder
PRE_SRC=bert-base-multilingual-uncased
PRE=bert-base-multilingual-uncased

#---------Battleship Setting-------------#
BATTLE="True"
GPU="-G"
NODE="s04"
CPU="8"
TIME="2-0"
MEM="40"

avg_topk_best_checkpoints(){
	python scripts/average_checkpoints.py \
	            --inputs $1 \
				--num-epoch-checkpoints $2 --output $3 \
				--ckpref checkpoints.best_bleu	
}

for No in "${CHECKPOINTS[@]}" ; do
	 
	CHECKPOINTS_ROOT=$CHECKPOINTS_PATH/$No
	
    avg_topk_best_checkpoints $CHECKPOINTS_ROOT $TOPK $CHECKPOINTS_ROOT/checkpoint_best_top$TOPK.pt

	# ---------------------------------------0
	ck_ch1=last
	ck_ch2=best
	for ck_ch in "${CHECK_TYPES[@]}"; do
	    for data_type in "${DATA_TYPES[@]}" ; do
			RESULT_PATH=$CHECKPOINTS_ROOT/$data_type/$ck_ch.bleu/$DATA
			CHECKPOINTS_DATA=checkpoint_$ck_ch.pt
				#hrun $GPU -N $NODE -c $CPU -t $TIME -m $MEM python generate.py \
				python generate.py \
					$DATABIN \
					--gen-subset $data_type \
					--task $TASK \
					--path $CHECKPOINTS_ROOT/$CHECKPOINTS_DATA \
					--results-path $RESULT_PATH \
					--arch $ARCH \
					--iter-decode-max-iter 0 \
					--criterion $CRITERION \
					--beam 1 \
					--no-repeat-ngram-size 1 \
					--left-pad-source \
                    --prepend-bos \
					--pretrained-lm-name bert-base-multilingual-uncased \
					--pretrained-model-name bert-base-multilingual-uncased \
					--sacrebleu \
					--bpe bibert \
					--pretrained-bpe ${PRE} --pretrained-bpe-src ${PRE_SRC} \
					--remove-bpe \
					--upsample-fill-mask \
					--batch-size $BATCH_SIZE 
			
		done
	done
done


			# --use-align-position \

	# 	python checkpoints/plotHT.py \
	# 		--data-path $RESULT_PATH \
	# 		--results-path $RESULT_PATH/$ck_ch.hit2d.png \
	# 		--clip --clip-y-min 0 --clip-y-max 50			
	# --hit2d \





# # # =============================================================================================================================
# CHECKPOINTS=("No-7-1-1-translation" "No-7-1-3-translation-lm")
# REORDER_TRANSLATION=translation
# # REORDER_TRANSLATION=reorder_translation
# # # DATA_TYPES=("test" "valid" "train")



# for No in "${CHECKPOINTS[@]}" ; do
	 
# 	CHECKPOINTS_ROOT=$CHECKPOINTS_PATH/$No
	
#     avg_topk_best_checkpoints $CHECKPOINTS_ROOT $TOPK $CHECKPOINTS_ROOT/checkpoint_best_top$TOPK.pt

# 	# ---------------------------------------0
# 	ck_ch1=last
# 	ck_ch2=best
# 	for ck_ch in "${CHECK_TYPES[@]}"; do
# 	    for data_type in "${DATA_TYPES[@]}" ; do
# 			RESULT_PATH=$CHECKPOINTS_ROOT/$data_type/$ck_ch.bleu/$DATA
# 			CHECKPOINTS_DATA=checkpoint_$ck_ch.pt
# 				#hrun $GPU -N $NODE -c $CPU -t $TIME -m $MEM python generate.py \
# 				python generate.py \
# 					$DATABIN \
# 					--gen-subset $data_type \
# 					--task $TASK \
# 					--path $CHECKPOINTS_ROOT/$CHECKPOINTS_DATA \
# 					--results-path $RESULT_PATH \
# 					--arch $ARCH \
# 					--iter-decode-max-iter 0 \
# 					--criterion $CRITERION \
# 					--beam 1 \
# 					--no-repeat-ngram-size 1 \
# 					--left-pad-source \
#                     --prepend-bos \
# 					--pretrained-lm-name bert-base-multilingual-uncased \
# 					--pretrained-model-name bert-base-multilingual-uncased \
# 					--pretrained-embedding-name bert-base-multilingual-uncased \
# 					--use-pretrained-embedding \
# 					--use-align-position \
# 					--sacrebleu \
# 					--bpe bibert \
# 					--pretrained-bpe ${PRE} --pretrained-bpe-src ${PRE_SRC} \
# 					--remove-bpe \
# 					--batch-size $BATCH_SIZE 
			
# 		done
# 	done
# done





