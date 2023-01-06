experiment_1=J-6-4-1-H12-UF20M
experiment_2=J-6-4-1-N-UF20M
# CUDA_VISIBLE_DEVICES=0 bash call_scripts/train_nat.sh -e $experiment_1 --twcc --fp16 --save-interval-updates 70000 --max-update 70000 --max-tokens 3072 -b 12288 -g 1 --dropout 0.1 --no-atten-mask
# mkdir checkpoints/$experiment_1/top5_700000steps
# cp checkpoints/$experiment_1/checkpoint.best_bleu_*  checkpoints/$experiment_1/top5_700000steps
# mkdir checkpoints/$experiment_2/
# cp checkpoints/$experiment_1/top5_700000steps/* checkpoints/$experiment_2/
# cp checkpoints/$experiment_1/checkpoint_last.pt checkpoints/$experiment_2/
CUDA_VISIBLE_DEVICES=0 bash call_scripts/train_nat.sh -e $experiment_1 --twcc --fp16 --save-interval-updates 70000 --max-update 100000 --max-tokens 3072 -b 12288 -g 1 --dropout 0.1 --no-atten-mask
CUDA_VISIBLE_DEVICES=0 bash call_scripts/train_nat.sh -e $experiment_2 --twcc --fp16 --save-interval-updates 70000 --max-update 100000 --max-tokens 3072 -b 12288 -g 1 --dropout 0.1 --no-atten-mask

CUDA_VISIBLE_DEVICES=0 bash call_scripts/generate_nat.sh -b 1 --twcc --data-subset test --ck-types top --avg-speed 1 --no-atten-mask \
                    -e $experiment_1 \
                    -e $experiment_2 \
#done