TRAIN_FILE=examples/iwslt14.tokenized.de-en/train.paste.de-en
EVAL_FILE=examples/iwslt14.tokenized.de-en/valid.paste.de-en
OUTPUT_DIR=output/train-finetune-9-2epoch
MODEL=output/train-finetune-5epoch
hrun -GG -N s03 -c 6 -t 5-0 -m 30 python run_train.py \
    --output_dir=$OUTPUT_DIR \
    --model_name_or_path=$MODEL \
    --extraction 'softmax' \
    --do_train \
    --train_mlm \
    --train_tlm \
    --train_tlm_full \
    --train_so \
    --train_psi \
    --train_data_file=$TRAIN_FILE \
    --per_gpu_train_batch_size 6 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --save_steps 5500 \
    --max_steps 1000 \
    --do_eval \
    --eval_data_file=$EVAL_FILE
