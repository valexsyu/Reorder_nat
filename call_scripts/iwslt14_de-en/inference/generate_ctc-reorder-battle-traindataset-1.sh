# --bpe bibert \
# --pretrained-bpe ${PRE} --pretrained-bpe-src ${PRE_SRC} \
# --pretrained-model-name jhu-clsp/bibert-ende \
# --use-align-position \
# -----------    Setting    -------------
CHECKPOINTS=("No-1-1-1-translation")
REORDER_TRANSLATION=translation
# REORDER_TRANSLATION=reorder_translation
# DATA_TYPES=("test" "valid" "train")
DATA_TYPES=("test")
CHECK_TYPES=("last")
DATA=iwslt14.de-en
#DATABIN=data/nat_position_reorder/awesome/Bibert_token_distill_iwslt14_de_en/de-en-databin #Bin Data of TEST dataset
DATABIN=data/nat_position_reorder/awesome/Bibert_token_distill_iwslt14_de_en_52k/de-en-databin #Bin Data of TEST dataset
# DATABIN=data/nat_position_reorder/awesome/Bibert_token_distill_iwslt14_de_en_mbert/de-en-databin #Bin Data of TEST dataset
BATCH_SIZE=50
ARCH=nat_position_reorder
#ARCH=nat_position_reorder_samll
CRITERION=nat_ctc_loss
TASK=translation_align_reorder
PRE_SRC=jhu-clsp/bibert-ende
PRE=jhu-clsp/bibert-ende

#---------Battleship Setting-------------#
BATTLE="True"
GPU="-G"
NODE="s04"
CPU="8"
TIME="2-0"
MEM="40"

CHECKPOINTS_PATH=checkpoints

# for No in "${CHECKPOINTS[@]}" ; do
	 
# 	CHECKPOINTS_ROOT=$CHECKPOINTS_PATH/$No
	
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
# 					--reorder-translation $REORDER_TRANSLATION \
# 					--beam 1 \
# 					--no-repeat-ngram-size 1 \
# 					--left-pad-source \
#                     --prepend-bos \
# 					--pretrained-lm-name jhu-clsp/bibert-ende \
# 					--pretrained-model-name jhu-clsp/bibert-ende \
# 					--pretrained-embedding-name jhu-clsp/bibert-ende \
# 					--use-pretrained-embedding \
# 					--sacrebleu \
# 					--bpe bibert \
# 					--pretrained-bpe ${PRE} --pretrained-bpe-src ${PRE_SRC} \
# 					--remove-bpe \
# 					--batch-size $BATCH_SIZE 
			
# 		done
# 	done
# done


			#--use-align-position \

		#python checkpoints/plotHT.py \
		#	--data-path $RESULT_PATH \
		#	--results-path $RESULT_PATH/$ck_ch.hit2d.png \
		#	--clip --clip-y-min 0 --clip-y-max 50			
	#--hit2d \





# # =============================================================================================================================
# CHECKPOINTS=("No-1-1-2-reorder-translation")
# # REORDER_TRANSLATION=translation
# REORDER_TRANSLATION=reorder_translation
# # DATA_TYPES=("test" "valid" "train")



# for No in "${CHECKPOINTS[@]}" ; do
	 
# 	CHECKPOINTS_ROOT=$CHECKPOINTS_PATH/$No
	
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
# 					--reorder-translation $REORDER_TRANSLATION \
# 					--beam 1 \
# 					--no-repeat-ngram-size 1 \
# 					--left-pad-source \
#                     --prepend-bos \
# 					--pretrained-lm-name jhu-clsp/bibert-ende \
# 					--pretrained-model-name jhu-clsp/bibert-ende \
# 					--pretrained-embedding-name jhu-clsp/bibert-ende \
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







# =============================================================================================================================
CHECKPOINTS=("No-1-1-4-reorder-translation-lm")
# REORDER_TRANSLATION=translation
REORDER_TRANSLATION=reorder_translation
# DATA_TYPES=("test" "valid" "train")


for No in "${CHECKPOINTS[@]}" ; do
	 
	CHECKPOINTS_ROOT=$CHECKPOINTS_PATH/$No
	
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
					--reorder-translation $REORDER_TRANSLATION \
					--beam 1 \
					--no-repeat-ngram-size 1 \
					--left-pad-source \
                    --prepend-bos \
					--pretrained-lm-name jhu-clsp/bibert-ende \
					--pretrained-model-name jhu-clsp/bibert-ende \
					--pretrained-embedding-name jhu-clsp/bibert-ende \
					--use-pretrained-embedding \
					--use-align-position \
					--sacrebleu \
					--bpe bibert \
					--pretrained-bpe ${PRE} --pretrained-bpe-src ${PRE_SRC} \
					--remove-bpe \
					--batch-size $BATCH_SIZE 
			
		done
	done
done


					# --post-process '##' \



# # =============================================================================================================================
# CHECKPOINTS=("No-test")
# # REORDER_TRANSLATION=translation
# REORDER_TRANSLATION=reorder_translation
# # DATA_TYPES=("test" "valid" "train")


# for No in "${CHECKPOINTS[@]}" ; do
	 
# 	CHECKPOINTS_ROOT=$CHECKPOINTS_PATH/$No
	
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
# 					--reorder-translation $REORDER_TRANSLATION \
# 					--beam 1 \
# 					--no-repeat-ngram-size 1 \
# 					--left-pad-source \
#                     --prepend-bos \
# 					--pretrained-lm-name jhu-clsp/bibert-ende \
# 					--pretrained-model-name jhu-clsp/bibert-ende \
# 					--pretrained-embedding-name jhu-clsp/bibert-ende \
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
# =============================================================================================================================




CHECKPOINTS=("No-1-1-00-translation")
REORDER_TRANSLATION=translation
# REORDER_TRANSLATION=reorder_translation
# DATA_TYPES=("test" "valid" "train")


for No in "${CHECKPOINTS[@]}" ; do
	 
	CHECKPOINTS_ROOT=$CHECKPOINTS_PATH/$No
	
	# ---------------------------------------0
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
				--reorder-translation $REORDER_TRANSLATION \
				--beam 1 \
				--no-repeat-ngram-size 1 \
				--left-pad-source \
				--prepend-bos \
				--pretrained-model-name jhu-clsp/bibert-ende \
				--sacrebleu \
				--bpe bibert \
				--pretrained-bpe ${PRE} --pretrained-bpe-src ${PRE_SRC} \
				--remove-bpe \
				--batch-size $BATCH_SIZE
			
		done
	done
done