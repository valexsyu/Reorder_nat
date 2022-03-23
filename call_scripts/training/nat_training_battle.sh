#!/bin/bash
#---------Path Setting-------------------#
CHECKPOINT=checkpoints/iwslt14.tokenized.de-en.No0-2
DATA_BIN=data-bin/iwslt14.tokenized.de-en
MAX_TOKENS=2500
MAX_EPOCH=600
CUR_START_EPOCH=300
CUDA_DEVICES=0,1
#---------Battleship Setting-------------#
BATTLE="False"
GPU="-G"
NODE="s04"
CPU="8"
TIME="3-0"
MEM="30"


#----------RUN  Bash-----------------------------2
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
    echo "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python train.py \\" >> $CHECKPOINT/temp.sh
fi

cat > $CHECKPOINT/temp1.sh << 'endmsg'
    $DATA_BIN \
    --save-dir $CHECKPOINT \
    --ddp-backend=legacy_ddp \
    --task translation_lev \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' \
    --share-all-embeddings \
    --dropout 0.3 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --log-format 'simple' --log-interval 100 \
    --fixed-validation-seed 7 \
    --save-interval-updates 10000 \
	--criterion nat_loss \
	--arch nonautoregressive_transformer \
    --noise random_mask \
    --iter-decode-max-iter 0 \
	--iter-decode-eos-penalty 0 \
    --iter-decode-force-max-iter \
    --tensorboard-logdir $CHECKPOINT/tensorboard \
    --no-epoch-checkpoints \
    --max-tokens $MAX_TOKENS \
    --max-epoch $MAX_EPOCH \
    --update-freq 4 \
    --apply-bert-init \
    --save-interval 1
#--curricular-learning \
#--curricular-learning-start-epoch $CUR_START_EPOCH \
#--curricular-learning-end-epoch $MAX_EPOCH \
#--apply-bert-init \
#--finetune-from-model $CHECKPOINT/checkpoint_last.pt \
#--reset-meters \
#--left-pad-source \
#--best-checkpoint-metric bleu \
#--maximize-best-checkpoint-metric \
#--eval-bleu \    

endmsg

cat $CHECKPOINT/temp.sh $CHECKPOINT/temp1.sh > $CHECKPOINT/scrip.sh
rm $CHECKPOINT/temp*

bash $CHECKPOINT/scrip.sh

:<<'END_COMMENT' 
END_COMMENT