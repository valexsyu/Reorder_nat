# -----------    Setting    -------------
#CHECKPOINTS=("No0-5" "No0-6" "No0-7" "No0-8")
CHECKPOINTS=("No-2")
DATA_TYPES=("train")  #("test" "valid" "train")
DATA=iwslt14.tokenized.de-en #Bin Data of TEST dataset  ########################### without distilled data
BATCH_SIZE=200
ARCH=nonautoregressive_reorder_translation
CRITERION=nat_ctc_loss
TASK=translation_encoder_only

#---------Battleship Setting-------------#
BATTLE="True"
GPU="-G"
NODE="s04"
CPU="8"
TIME="2-0"
MEM="30"

CHECKPOINTS_PATH=checkpoints

for No in "${CHECKPOINTS[@]}" ; do
	 
	CHECKPOINTS_ROOT=$CHECKPOINTS_PATH/$No
	
	# ---------------------------------------0
	ck_ch1=last
	ck_ch2=best
	for ck_ch in $ck_ch1 $ck_ch2; do
	    for data_type in "${DATA_TYPES[@]}" ; do
			RESULT_PATH=$CHECKPOINTS_ROOT/$data_type/$ck_ch.bleu/$DATA
			CHECKPOINTS_DATA=checkpoint_$ck_ch.pt
				hrun -s $GPU -N $NODE -c $CPU -t $TIME -m $MEM python generate.py \
					data-bin/$DATA \
					--gen-subset $data_type \
					--task $TASK \
					--path $CHECKPOINTS_ROOT/$CHECKPOINTS_DATA \
					--remove-bpe \
					--results-path $RESULT_PATH \
					--arch $ARCH \
					--iter-decode-max-iter 0 \
					--criterion $CRITERION \
					--reorder-translation reorder_translation \
					--beam 1 \
					--no-repeat-ngram-size 1 \
					--batch-size $BATCH_SIZE
		done
	done
done


			#--iter-decode-max-iter 0 \
			#--iter-decode-eos-penalty 0 \
			#--iter-decode-force-max-iter \
			#--hist2d-only \
			#--beam 1

		#python checkpoints/plotHT.py \
		#	--data-path $RESULT_PATH \
		#	--results-path $RESULT_PATH/$ck_ch.hit2d.png \
		#	--clip --clip-y-min 0 --clip-y-max 50			
	#--hit2d \















