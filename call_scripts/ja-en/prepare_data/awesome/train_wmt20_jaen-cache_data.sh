TRAIN_FILE=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/wmt20_jaen_detoken_non-distilled/train.paste.ja-en
EVAL_FILE=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/wmt20_jaen_detoken_non-distilled/valid.paste.ja-en
OUTPUT_DIR=output/finetune-awesome-jaen-2-epoch-cache_data
MODEL=/home/valexsyu/Doc/NMT/awesome-align/output/awesome_trained/model_with_co
awesome-train \
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
    --save_steps 5000 \
    --max_steps 60000 \
    --do_eval \
    --cache_data \
    --eval_data_file=$EVAL_FILE


    # --should_continue \
    # --cache_data \
    # --overwrite_output_dir \

    # --should_continue \


