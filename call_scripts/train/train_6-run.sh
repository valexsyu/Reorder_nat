source $HOME/.bashrc 
conda activate base
# bash call_scripts/train_nat.sh -e  b-2-1-1-H12-UR40M --fp16 -g 1 --save-interval-updates 10000 --max-update 100000 \
#                                                      --lm-start-step 75000 --max-tokens 2048 -b 65536 --no-atten-mask --has-eos
# hrun -s -N s01 -GGGG -c 12 -m 30 -t 3-0 bash call_scripts/train_nat.sh \
#         -e  2-2-1-1-H12-UF20T-eos --fp16 -g 4 --save-interval-updates 70000 --max-update 100000 \
#         --lm-start-step 75000 --max-tokens 1024 -b 12288 --no-atten-mask --has-eos

# experiment_1=2-2-1-1-H12-UF20T-eos 
# experiment_2=2-2-1-1-N-UF20T-eos 
# hrun -s -N s01 -GGGG -c 4 -m 30 -t 3-0 bash call_scripts/train_nat.sh \
#         -e  $experiment_1 --fp16 -g 4 --save-interval-updates 70000 --max-update 70000 \
#         --lm-start-step 75000 --max-tokens 1024 -b 12288 --no-atten-mask --has-eos
# mkdir checkpoints/$experiment_1/top5_700000steps    
# cp checkpoints/$experiment_1/checkpoint.best_bleu_*  checkpoints/$experiment_1/top5_700000steps
# mkdir checkpoints/$experiment_2/
# cp checkpoints/$experiment_1/top5_700000steps/* checkpoints/$experiment_2/
# cp checkpoints/$experiment_1/checkpoint_last.pt checkpoints/$experiment_2/
# hrun -s -N s01 -GGGG -c 4 -m 30 -t 3-0 bash call_scripts/train_nat.sh \
#         -e  $experiment_1 --fp16 -g 4 --save-interval-updates 70000 --max-update 100000 \
#         --lm-start-step 75000 --max-tokens 1024 -b 12288 --no-atten-mask --has-eos
# hrun -s -N s01 -GGGG -c 4 -m 30 -t 3-0 bash call_scripts/train_nat.sh \
#         -e  $experiment_2 --fp16 -g 4 --save-interval-updates 70000 --max-update 100000 \
#         --lm-start-step 75000 --max-tokens 1024 -b 12288 --no-atten-mask --has-eos    



# hrun -s -N s01 -GGGG -c 12 -m 30 -t 3-0 bash call_scripts/train_nat.sh \
#         -e  2-2-2-1-H12-UF20T-eos --fp16 -g 4 --save-interval-updates 70000 --max-update 100000 \
#         --lm-start-step 75000 --max-tokens 1024 -b 12288 --no-atten-mask --has-eos        


experiments=( "1-1-1-1-H12-UF13T" "2-2-1-1-H12-UF13T" "5-3-1-1-H12-UF13T" "7-4-1-1-H12-UF13T" )
for experiment in "${experiments[@]}" ; do 
        bash call_scripts/train_nat.sh \
                -e  $experiment --fp16 -g 1 --save-interval-updates 70000 --max-update 100000 \
                --lm-start-step 75000 --max-tokens 1024 -b 12288 --no-atten-mask --has-eos   
done                