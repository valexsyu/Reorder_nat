#---------Path Setting-------------------#
CHECKPOINT=checkpoints/test_gg
DATA_BIN=data-bin/iwslt14.tokenized.de-en.distilled
MAX_TOKENS=2500
MAX_EPOCH=150
CUR_START_EPOCH=300
CUDA_DEVICES=0
#---------Battleship Setting-------------#
BATTLE="False"
GPU="-GGG"
NODE="s04"
CPU="4"
TIME="2-0"
MEM="30"


#----------RUN  Bash-----------------------------
mkdir $CHECKPOINT
mkdir $CHECKPOINT/tensorboard
echo "
CHECKPOINT=$CHECKPOINT
DATA_BIN=$DATA_BIN
MAX_TOKENS=$MAX_TOKENS
MAX_EPOCH=$MAX_EPOCH
CUR_START_EPOCH=$CUR_START_EPOCH
"  > $CHECKPOINT/temp.sh

if [ $BATTLE == "True" ] ; then
    echo "hrun $GPU -N $NODE -c $CPU -t $TIME -m $MEM  python train.py \\" >> $CHECKPOINT/temp.sh
else
    echo "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES  python train.py \\" >> $CHECKPOINT/temp.sh
fi

cat > $CHECKPOINT/temp1.sh << 'endmsg'
    $DATA_BIN \
    --save-dir $CHECKPOINT \
    --ddp-backend=legacy_ddp \
    --task translation_encoder_only \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' \
    --weight-decay 0.01 \
    --log-format 'simple' --log-interval 100 \
    --fixed-validation-seed 7 \
    --save-interval-updates 10000 \
	--criterion nat_ctc_loss \
	--arch nonautoregressive_roberta \
    --tensorboard-logdir $CHECKPOINT/tensorboard \
    --no-epoch-checkpoints \
    --max-tokens $MAX_TOKENS \
    --max-epoch $MAX_EPOCH \
    --max-source-positions 1024 \
    --max-target-positions 1024 \
    --update-freq 4 \
    --noise random_mask \
    --num-upsampling-rate 3 \
    --save-interval 1
#--curricular-learning \
#--curricular-learning-start-epoch $CUR_START_EPOCH \
#--curricular-learning-end-epoch $MAX_EPOCH \
#--curricular-learning-bleu \
#--curricular-learning-bleu-easytohard \
#--apply-bert-init \    
#--finetune-from-model $CHECKPOINT/checkpoint_last.pt \
#--reset-meters \
#--left-pad-source \
#--best-checkpoint-metric bleu \
#--maximize-best-checkpoint-metric \
#--eval-bleu \    
#--iter-decode-max-iter 0 \
#--iter-decode-eos-penalty 0 \     
#--iter-decode-force-max-iter \
#--noise random_mask \
#--share-all-embeddings \
#--dropout 0.3 
#--decoder-learned-pos \
#--encoder-learned-pos \
#--random-mask-rate 0 \
#--num-upsampling-rate 3 \
#--encoder-causal-attn \
#--reorder-translation reorder \
endmsg

cat $CHECKPOINT/temp.sh $CHECKPOINT/temp1.sh > $CHECKPOINT/scrip.sh
rm $CHECKPOINT/temp*

bash $CHECKPOINT/scrip.sh




:<<'END_COMMENT' 
	#--criterion nat_loss --label-smoothing 0.1 \
	#--clip-norm 0.05 \
	#data-bin/wmt14_en_de_distill \
	#--save-dir checkpoints/wmt14_en_de_distill \
	#--best-checkpoint-metric bleu \
    #--feature-second-loss \
    #--feature-first-loss \
    #--src-upsample-rate 3
    #data-bin/iwslt14.tokenized.de-en.distilled.de-reorder-finetune \
    #--src-embedding-copy \
    #--noise random_mask\89
    # 3500 2080*2
    #data-bin/iwslt14.tokenized.de-en.distlled.de-reorder-train3 \
    #--save-dir checkpoints/iwslt14.tokenized.de-en.distilled.de-reorder-train3.noise \

    #data-bin/iwslt14.tokenized.de-en.distilled \  data-bin/iwslt14.tokenized.de-en.distilled-small \
	#--save-dir checkpoints/iwslt14.tokenized.de-en.distilled \
    #--eval-bleu-detok moses \
    #--eval-bleu-remove-bpe \
    #--eval-tokenized-bleu \
    #--reset-meters \
    #--best-checkpoint-metric bleu \
    #--maximize-best-checkpoint-metric \
    #--eval-bleu \
    #--iter-decode-max-iter 0 \
	#--iter-decode-eos-penalty 0 \
    #--iter-decode-force-max-iter \  
    #--reset-meters \  
    --left-pad-source \