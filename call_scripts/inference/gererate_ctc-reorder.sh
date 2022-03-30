# -----------    Setting    -------------
#CHECKPOINTS=("No0-5" "No0-6" "No0-7" "No0-8")
BATTLE="False"
CHECKPOINTS=("testQQ")
DATA=iwslt14.tokenized.de-en.distilled #Bin Data of TEST dataset
DEVICE=0,1
BATCH_SIZE=30
ARCH=nonautoregressive_roberta
CRITERION=nat_ctc_loss
TASK=translation_encoder_only

if [ $BATTLE == "True" ] ; then
    CHECKPOINTS_PATH=checkpoints/battle
else
    CHECKPOINTS_PATH=checkpoints
fi

for No in "${CHECKPOINTS[@]}" ; do
	 
	CHECKPOINTS_ROOT=$CHECKPOINTS_PATH/$No
	
	# ---------------------------------------0
	ck_ch1=last
	ck_ch2=best
	for ck_ch in $ck_ch1 $ck_ch2; do
	RESULT_PATH=$CHECKPOINTS_ROOT/$ck_ch.bleu/$DATA
	CHECKPOINTS_DATA=checkpoint_$ck_ch.pt
		CUDA_VISIBLE_DEVICES=$DEVICE CUDA_LAUNCH_BLOCKING=1 python generate.py \
			data-bin/$DATA \
			--gen-subset test \
			--task $TASK \
			--path $CHECKPOINTS_ROOT/$CHECKPOINTS_DATA \
			--remove-bpe \
			--results-path $RESULT_PATH \
			--arch $ARCH \
			--iter-decode-max-iter 0 \
			--criterion $CRITERION \
			--batch-size $BATCH_SIZE
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















