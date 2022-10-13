source $HOME/.bashrc 
conda activate base

#---------Path Setting-------------------#
CHECKPOINT=checkpoints/No-99-transformer-iwslt14-bibert
DATA_BIN=data/nat_position_reorder/awesome/Bibert_token_iwslt14_de_en_52k/de-en-databin
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
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --max-update 100000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 12288 \
    --wandb-project ReorderNAT99 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric



endmsg

cat $CHECKPOINT/temp.sh $CHECKPOINT/temp1.sh > $CHECKPOINT/scrip.sh
rm $CHECKPOINT/temp*

bash $CHECKPOINT/scrip.sh




:<<'END_COMMENT' 
	