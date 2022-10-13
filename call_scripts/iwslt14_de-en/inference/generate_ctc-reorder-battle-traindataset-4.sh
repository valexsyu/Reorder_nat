# --pretrained-bpe ${PRE} --pretrained-bpe-src ${PRE_SRC} \
# --use-align-position \
# --bpe bibert \


# -----------    Setting    -------------
CHECKPOINTS=("No-2-1-1-translation")
REORDER_TRANSLATION=translation
# REORDER_TRANSLATION=reorder_translation
# DATA_TYPES=("test" "valid" "train")
DATA_TYPES=("test")
DATA=wmt14.de-en  
#DATABIN=data/nat_position_reorder/awesome/Bibert_token_distill_iwslt14_de_en/de-en-databin #Bin Data of TEST dataset
# DATABIN=data/nat_position_reorder/awesome/Bibert_token_distill_iwslt14_de_en_52k/de-en-databin #Bin Data of TEST dataset
# DATABIN=data/nat_position_reorder/awesome/Bibert_token_distill_iwslt14_de_en_mbert/de-en-databin #Bin Data of TEST dataset
# DATABIN=data/nat_position_reorder/awesome/Bibert_token_distill_iwslt14_de_en_dual-bert-de-en/de-en-databin #Bin Data of TEST dataset
DATABIN=data/nat_position_reorder/awesome/Bibert_token_distill_wmt14_de_en_mbert/de-en-databin
BATCH_SIZE=30
ARCH=nat_position_reorder
#ARCH=nat_position_reorder_samll
CRITERION=nat_ctc_loss
TASK=translation_align_reorder
PRE_SRC=jhu-clsp/bibert-ende
PRE=data/nat_position_reorder/bibert/8k-vocab-models

#---------Battleship Setting-------------#
BATTLE="True"
GPU="-G"
NODE="s04"
CPU="8"
TIME="2-0"
MEM="40"

CHECKPOINTS_PATH=checkpoints

for No in "${CHECKPOINTS[@]}" ; do
	 
	CHECKPOINTS_ROOT=$CHECKPOINTS_PATH/$No
	
	# ---------------------------------------0
	ck_ch1=last
	ck_ch2=best
	for ck_ch in $ck_ch1 $ck_ch2 ; do
	    for data_type in "${DATA_TYPES[@]}" ; do
			RESULT_PATH=$CHECKPOINTS_ROOT/$data_type/$ck_ch.bleu/$DATA
			CHECKPOINTS_DATA=checkpoint_$ck_ch.pt
				#hrun $GPU -N $NODE -c $CPU -t $TIME -m $MEM python generate.py \
				python generate.py \
					$DATABIN \
					--gen-subset $data_type \
					--task $TASK \
					--path $CHECKPOINTS_ROOT/$CHECKPOINTS_DATA \
					--remove-bpe \
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
					--sacrebleu \
					--batch-size $BATCH_SIZE 
			
			bash call_scripts/inference/get_entropy.sh $RESULT_PATH/generate-$data_type.txt $RESULT_PATH/generate-$data_type-entropy.txt $data_type
		done
	done
done


# ========================================================================================================================


# -----------    Setting    -------------
CHECKPOINTS=("No-2-1-2-reorder-translation")
# REORDER_TRANSLATION=translation
REORDER_TRANSLATION=reorder_translation
DATA_TYPES=("test")
DATA=wmt14.de-en  
BATCH_SIZE=30
for No in "${CHECKPOINTS[@]}" ; do
	 
	CHECKPOINTS_ROOT=$CHECKPOINTS_PATH/$No
	
	# ---------------------------------------0
	ck_ch1=last
	ck_ch2=best
	for ck_ch in $ck_ch1 $ck_ch2 ; do
	    for data_type in "${DATA_TYPES[@]}" ; do
			RESULT_PATH=$CHECKPOINTS_ROOT/$data_type/$ck_ch.bleu/$DATA
			CHECKPOINTS_DATA=checkpoint_$ck_ch.pt
				#hrun $GPU -N $NODE -c $CPU -t $TIME -m $MEM python generate.py \
				python generate.py \
					$DATABIN \
					--gen-subset $data_type \
					--task $TASK \
					--path $CHECKPOINTS_ROOT/$CHECKPOINTS_DATA \
					--remove-bpe \
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
					--batch-size $BATCH_SIZE 
			
			bash call_scripts/inference/get_entropy.sh $RESULT_PATH/generate-$data_type.txt $RESULT_PATH/generate-$data_type-entropy.txt $data_type
		done
	done
done


# ========================================================================================================================


# -----------    Setting    -------------
CHECKPOINTS=("No-2-1-3-translation-lm")
REORDER_TRANSLATION=translation
# REORDER_TRANSLATION=reorder_translation
DATA_TYPES=("test")
DATA=wmt14.de-en  
BATCH_SIZE=30
for No in "${CHECKPOINTS[@]}" ; do
	 
	CHECKPOINTS_ROOT=$CHECKPOINTS_PATH/$No
	
	# ---------------------------------------0
	ck_ch1=last
	ck_ch2=best
	for ck_ch in $ck_ch1 $ck_ch2 ; do
	    for data_type in "${DATA_TYPES[@]}" ; do
			RESULT_PATH=$CHECKPOINTS_ROOT/$data_type/$ck_ch.bleu/$DATA
			CHECKPOINTS_DATA=checkpoint_$ck_ch.pt
				#hrun $GPU -N $NODE -c $CPU -t $TIME -m $MEM python generate.py \
				python generate.py \
					$DATABIN \
					--gen-subset $data_type \
					--task $TASK \
					--path $CHECKPOINTS_ROOT/$CHECKPOINTS_DATA \
					--remove-bpe \
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
					--sacrebleu \
					--batch-size $BATCH_SIZE 
			
			bash call_scripts/inference/get_entropy.sh $RESULT_PATH/generate-$data_type.txt $RESULT_PATH/generate-$data_type-entropy.txt $data_type
		done
	done
done


# ========================================================================================================================


# -----------    Setting    -------------
CHECKPOINTS=("No-2-1-4-reorder-translation-lm")
# REORDER_TRANSLATION=translation
REORDER_TRANSLATION=reorder_translation
DATA_TYPES=("test")
DATA=wmt14.de-en  
BATCH_SIZE=30
for No in "${CHECKPOINTS[@]}" ; do
	 
	CHECKPOINTS_ROOT=$CHECKPOINTS_PATH/$No
	
	# ---------------------------------------0
	ck_ch1=last
	ck_ch2=best
	for ck_ch in $ck_ch1 $ck_ch2 ; do
	    for data_type in "${DATA_TYPES[@]}" ; do
			RESULT_PATH=$CHECKPOINTS_ROOT/$data_type/$ck_ch.bleu/$DATA
			CHECKPOINTS_DATA=checkpoint_$ck_ch.pt
				#hrun $GPU -N $NODE -c $CPU -t $TIME -m $MEM python generate.py \
				python generate.py \
					$DATABIN \
					--gen-subset $data_type \
					--task $TASK \
					--path $CHECKPOINTS_ROOT/$CHECKPOINTS_DATA \
					--remove-bpe \
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
					--batch-size $BATCH_SIZE 
			
			bash call_scripts/inference/get_entropy.sh $RESULT_PATH/generate-$data_type.txt $RESULT_PATH/generate-$data_type-entropy.txt $data_type
		done
	done
done















