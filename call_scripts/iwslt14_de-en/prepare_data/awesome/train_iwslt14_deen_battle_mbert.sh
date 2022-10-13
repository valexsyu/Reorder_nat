TRAIN_FILE=examples/Bibert_detoken_iwslt14_de_en/train.paste.de-en
EVAL_FILE=examples/Bibert_detoken_iwslt14_de_en/valid.paste.de-en
OUTPUT_DIR=output/finetune-mbert-2-epoch
#MODEL=output/train-finetune-5epoch
MODEL=bert-base-multilingual-cased
CUDA_VISIBLE_DEVICES=0,1,2 python run_train.py \
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
    --per_gpu_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --save_steps 10000 \
    --max_steps 40000 \
    --do_eval \
    --eval_data_file=$EVAL_FILE
