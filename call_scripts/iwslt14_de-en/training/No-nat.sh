source $HOME/.bashrc 
conda activate base
#---------Path Setting-------------------#
CHECKPOINT=checkpoints/No-test
DATA_BIN=data/nat_position_reorder/awesome/Bibert_token_distill_baseline_iwslt14_de_en_52k/de-en-databin
MAX_TOKENS=2048
MAX_EPOCH=400
CUR_START_EPOCH=300
CUDA_DEVICES=0
#---------Battleship Setting-------------#
BATTLE="True"
GPU="-G"
NODE="s04"
CPU="10"
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
    echo "python train.py \\" >> $CHECKPOINT/temp.sh
else
    echo "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES  python train.py \\" >> $CHECKPOINT/temp.sh
fi

cat > $CHECKPOINT/temp1.sh << 'endmsg'
    $DATA_BIN \
    --save-dir $CHECKPOINT \
    --ddp-backend=legacy_ddp \
    --task translation_align_reorder \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0002 --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' \
    --weight-decay 0.01 \
    --log-format 'simple' --log-interval 100 \
    --fixed-validation-seed 7 \
    --save-interval-updates 10000 \
	--criterion nat_ctc_loss \
	--arch nat_pretrained_model \
    --tensorboard-logdir $CHECKPOINT/tensorboard \
    --no-epoch-checkpoints \
    --max-tokens $MAX_TOKENS \
    --max-epoch $MAX_EPOCH \
    --noise no_noise \
    --num-upsampling-rate 2 \
    --save-interval 1 \
    --left-pad-source \
    --prepend-bos \
    --align-position-pad-index 513 \
    --update-freq 6 \
    --keep-best-checkpoints 5 \
    --wandb-project Test \
    --wandb-entity valex-jcx \
    --pretrained-model-name jhu-clsp/bibert-ende \
    --pretrained-lm-name jhu-clsp/bibert-ende \
    --eval-bleu-print-samples \
    --eval-bleu --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --max-update 100000 \
    --lm-head-frozen \
    --upsample-fill-mask \
    --train-subset valid
endmsg

cat $CHECKPOINT/temp.sh $CHECKPOINT/temp1.sh > $CHECKPOINT/scrip.sh
rm $CHECKPOINT/temp*

bash $CHECKPOINT/scrip.sh

:<<'END_COMMENT' 

    # --pretrained-lm-name jhu-clsp/bibert-ende \


