function avg_topk_best_checkpoints(){
	python scripts/average_checkpoints.py \
	            --inputs $1 \
				--num-epoch-checkpoints $2 --output $3 \
				--ckpref checkpoints.best_bleu	
}

function avg_lastk_checkpoints(){
	python scripts/average_checkpoints.py \
	            --inputs $1 \
				--num-epoch-checkpoints $2 --output $3
}

CHECKPOINT=checkpoints/$1
TOPK=5

while :
do
    avg_topk_best_checkpoints $CHECKPOINT $TOPK $CHECKPOINT/checkpoint_best_top$TOPK.pt
    avg_lastk_checkpoints $CHECKPOINT $TOPK $CHECKPOINT/checkpoint_last$TOPK.pt

    date
    echo "Sleep $2 Now"
    sleep $2
done