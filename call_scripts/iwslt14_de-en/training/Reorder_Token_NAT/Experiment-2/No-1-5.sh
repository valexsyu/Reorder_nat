#---------Path Setting-------------------#
CHECKPOINT=checkpoints/No-1-5
DATA_BIN=data-bin/iwslt14.tokenized.de-en.distilled
MAX_TOKENS=6200
MAX_EPOCH=200
CUR_START_EPOCH=300
CUDA_DEVICES=0
#---------Battleship Setting-------------#
BATTLE="True"
GPU="-GGG"
NODE="s04"
CPU="8"
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
    echo "hrun -x $GPU -N $NODE -c $CPU -t $TIME -m $MEM  python train.py \\" >> $CHECKPOINT/temp.sh
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
	--arch nonautoregressive_reorder_translation \
    --tensorboard-logdir $CHECKPOINT/tensorboard \
    --no-epoch-checkpoints \
    --max-tokens $MAX_TOKENS \
    --max-epoch $MAX_EPOCH \
    --noise no_noise \
    --num-upsampling-rate 3 \
    --reorder-translation reorder_translation \
    --encoder-causal-attn \
    --pretrained-translation checkpoints/No-0-2-translation/checkpoint_last.pt \
    --pretrained-reorder checkpoints/No-0-1-reorder/checkpoint_last.pt \
    --wandb-project ReorderNAT2-No-1-5 \
    --freeze-module translator \
    --global-token \
    --left-pad-source \
    --prepend-bos \
    --add-blank-symbol \
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
#--update-freq 4 \
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
    #--left-pad-source \
