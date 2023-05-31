source $HOME/.bashrc 
conda activate base

# bash call_scripts/train_nat.sh -e E-2-1-1-H4-UR40M --fp16 -g 4 --save-interval-updates 10000 --max-update 130000 --lm-start-step 130000 --max-tokens 1638 -b 65520 --no-atten-mask

#bash call_scripts/train_nat.sh -e O-2-1-1-H12-UD25M --fp16 -g 4 --save-interval-updates 32500 --max-update 200000 --lm-start-step 130000 --max-tokens 2048 -b 65536 --no-atten-mask


# bash call_scripts/train_nat.sh -e G-2-1-1-H12-UD25M --fp16 -g 4 --save-interval-updates 32500 --max-update 200000 --lm-start-step 130000 --max-tokens 2048 -b 65536 --no-atten-mask
# bash call_scripts/train_nat.sh -e N-6-1-1-H12-UD25M --fp16 -g 4 --save-interval-updates 32500 --max-update 200000 --lm-start-step 130000 --max-tokens 2048 -b 65536 --no-atten-mask
# bash call_scripts/train_nat.sh -e G-2-1-1-H12-UR40M --fp16 -g 4 --save-interval-updates 32500 --max-update 200000 --lm-start-step 130000 --max-tokens 1638 -b 65520 --no-atten-mask  ##65000
# bashs
# bash ../BiBert/train-wmt-one-way-12k_bibertenv.sh
# sleep 10
# bash call_scripts/train_nat.sh -e  T-6-1-1-N-UF30T --fp16 -g 4 --save-interval-updates 32500 --max-update 200000 --lm-start-step 130000 --max-tokens 2048 -b 65536 --no-atten-mask --has-eos

# bash call_scripts/train_nat.sh -e  S-6-1-1-N-UF30T --fp16 -g 4 --save-interval-updates 32500 --max-update 200000 --lm-start-step 130000 --max-tokens 2048 -b 65520 --no-atten-mask --has-eos
# bash call_scripts/train_nat.sh -e  T-2-1-1-N-UF30T --fp16 -g 4 --save-interval-updates 32500 --max-update 200000 --lm-start-step 130000 --max-tokens 2048 -b 65536 --no-atten-mask --has-eos
# Stop bash call_scripts/train_nat.sh -e  T-2-1-1-N-UR40T --fp16 -g 2 --save-interval-updates 32500 --max-update 200000 --lm-start-step 130000 --max-tokens 2048 -b 65536 --no-atten-mask --has-eos
# bash call_scripts/train_nat.sh -e  Y-2-1-1-N-UR40T --fp16 -g 2 --save-interval-updates 10000 --max-update 100000 --lm-start-step 75000 --max-tokens 2048 -b 65536 --no-atten-mask --has-eos
# bash call_scripts/train_nat.sh -e  Y-6-1-1-N-UF30T --fp16 -g 2 --save-interval-updates 10000 --max-update 100000 --lm-start-step 75000 --max-tokens 2048 -b 65536 --no-atten-mask --has-eos
# bash call_scripts/train_nat.sh -e  Z-6-1-1-N-UF30T --fp16 -g 2 --save-interval-updates 10000 --max-update 100000 --lm-start-step 75000 --max-tokens 2048 -b 65536 --no-atten-mask --has-eos
# bash call_scripts/train_nat.sh -e  a-2-1-1-H12-UR40M --fp16 -g 2 --save-interval-updates 10000 --max-update 100000 \
#                                                      --lm-start-step 75000 --max-tokens 2048 -b 65536 --no-atten-mask --has-eos
# bash call_scripts/train_nat.sh -e L-1-1-1-H12-UR40M -g 1 --max-update 30000 --save-interval-updates 100000 --max-tokens 2048 -b 65536 \
#                                                     --fp16 --lm-start-step 20000 --dropout 0.1 --warmup-updates 3000 --no-atten-mask --has-eos \

# function pair_experiment() {
#     experiment_1=$1
#     experiment_2=$2
#     bash call_scripts/train_nat.sh \
#             -e $experiment_1 --fp16 -g 2 \
#             --save-interval-updates 70000 --max-update 70000 --lm-start-step 75000 \
#             --max-tokens 1024 -b 65536 --no-atten-mask 
#     mkdir checkpoints/$experiment_1/top5_70000steps    
#     cp checkpoints/$experiment_1/checkpoint.best_bleu_*  checkpoints/$experiment_1/top5_70000steps
#     mkdir checkpoints/$experiment_2/
#     cp checkpoints/$experiment_1/top5_70000steps/* checkpoints/$experiment_2/
#     cp checkpoints/$experiment_1/checkpoint_last.pt checkpoints/$experiment_2/
#     bash call_scripts/train_nat.sh \
#             -e $experiment_1 --fp16 -g 2 \
#             --save-interval-updates 70000 --max-update 100000 --lm-start-step 75000 \
#             --max-tokens 1024 -b 65536 --no-atten-mask 
# }

# pair_experiment h-7-3-1-H12-UD25M h-7-3-1-N-UD25M


# bash call_scripts/train_nat.sh -e 1-1-1-1-H12-UF20M-P2 --fp16 -g 1 \
#                                --save-interval-updates 50000 --max-tokens 2048 --has-eos --max-update 75000 --lm-start-step 55000 
# bash call_scripts/generate_nat.sh -b 40 --data-subset test --avg-speed 1 \
#                                -e 1-1-1-1-H12-UF20M-P2


# bash call_scripts/train_nat.sh -e m-8-1-1-K12-UF20M-AutoModelForMaskedLM  \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16

# bash call_scripts/train_nat.sh -e m-8-4-1-K12-UF20M-AutoModelForMaskedLM-randPos  \
#                                --save-interval-updates 70000 --max-tokens 4096 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16


# bash call_scripts/train_nat.sh -e m-8-2-1-K12-UF20M-AutoModelForMaskedLM-randPos  \
#                                --save-interval-updates 70000 --max-tokens 4096 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16


# bash call_scripts/train_nat.sh -e m-8-4-1-K12-UF20M-AutoModelForMaskedLM \
#                                --save-interval-updates 70000 --max-tokens 4096 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16


# bash call_scripts/train_nat.sh -e m-8-2-1-K12-UR40M-AutoModelForMaskedLM-randPos  \
#                                --save-interval-updates 70000 --max-tokens 3072 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16


# bash call_scripts/train_nat.sh -e m-8-2-3-K12-UF20B \
#                                --save-interval-updates 70000 --max-tokens 4096 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16



# bash call_scripts/train_nat.sh -e m-8-2-4-K12-UF20B \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16

# bash call_scripts/train_nat.sh -e m-8-1-4-K12-UR40B \
#                                --save-interval-updates 70000 --max-tokens 3072 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16


# bash call_scripts/train_nat.sh -e m-8-4-1-K12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16      


# bash call_scripts/train_nat.sh -e m-8-3-1-H12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16      

# bash call_scripts/train_nat.sh -e m-8-1-1-H12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16   

# bash call_scripts/train_nat.sh -e m-8-2-1-H12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16 

# bash call_scripts/train_nat.sh -e m-8-4-1-H12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16        


# bash call_scripts/train_nat.sh -e m-8-3-1-H12-UF20M-dp001 \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16 --dropout 0.01   

# bash call_scripts/train_nat.sh -e m-8-3-1-K12-UF20M-dp001 \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16 --dropout 0.01  

# bash call_scripts/train_nat.sh -e m-8-3-3-K12-UF20M-dp001-HDonly \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16 --dropout 0.2


# bash call_scripts/train_nat.sh -e m-8-3-3-K12-UF20M-dp02-HDonly \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16 --dropout 0.2

# #10
# bash call_scripts/train_nat.sh -e m-8-3-4-K12-UF20M-dp001 \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16 --dropout 0.01

# #11
# bash call_scripts/train_nat.sh -e m-8-3-3-K12-UF20M-dp02 \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16 --dropout 0.2



# #12
# bash call_scripts/train_nat.sh -e m-8-3-1-H12-UF20M-dp001 \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16 --dropout 0.01 

# #13
# bash call_scripts/train_nat.sh -e m-8-1-3-H12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16


# #14
# bash call_scripts/train_nat.sh -e m-8-2-3-H12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16 


# #15
# bash call_scripts/train_nat.sh -e m-8-3-3-H12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16 

# #16
# bash call_scripts/train_nat.sh -e m-8-4-3-H12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16                                


# #1
# bash call_scripts/train_nat.sh -e m-8-1-3-A12-UF20M \
#                                 --save-interval-updates 70000 --max-tokens 6144 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 -g 2 --fp16                                


# #2
# bash call_scripts/train_nat.sh -e m-8-3-3-A12-UF20M \
#                                 --save-interval-updates 70000 --max-tokens 6144 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 -g 2 --fp16                                



# #3
# bash call_scripts/train_nat.sh -e m-B-3-1-B12-UF20M-lm5 \
#                             --save-interval-updates 70000 --max-tokens 3072 \
#                             --has-eos --max-update 100000 --lm-start-step 75000 \
#                             --lm-iter-num 5 --watch-lm-loss \
#                             -g 2 --fp16 

# #4
# bash call_scripts/train_nat.sh -e m-B-1-3-B12-UF20B \
#                             --save-interval-updates 70000 --max-tokens 6144 \
#                             --has-eos --max-update 100000 --lm-start-step 75000 \
#                             --lm-iter-num 1 \
#                             -g 2 --fp16 


# #5
# bash call_scripts/train_nat.sh -e m-B-3-1-C12-UF20M-lm5x015 \
#                                 --save-interval-updates 70000 --max-tokens 3072 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 --lm-iter-num 5 --lm-mask-rate 0.15 \
#                                 -g 2 --fp16 

# #6
# bash call_scripts/train_nat.sh -e m-B-3-1-A12-UF20M-lmx015 \
#                                 --save-interval-updates 70000 --max-tokens 6144 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 --lm-mask-rate 0.15 \
#                                 -g 2 --fp16                                 

# #  #7
# bash call_scripts/train_nat.sh -e m-B-3-1-A12-UF20M-lm5x015 \
#                                 --save-interval-updates 70000 --max-tokens 6144 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 --lm-iter-num 5 --lm-mask-rate 0.15 \
#                                 -g 2 --fp16 

# #8
# bash call_scripts/train_nat.sh -e m-B-3-3-B12-UF20M-lmx015 \
#                                 --save-interval-updates 70000 --max-tokens 6144 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 --lm-mask-rate 0.15 \
#                                 -g 2 --fp16                                 

# #9
# bash call_scripts/train_nat.sh -e m-B-3-3-B12-UF20B-lmx015 \
#                                 --save-interval-updates 70000 --max-tokens 6144 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 --lm-mask-rate 0.15 \
#                                 -g 2 --fp16          



# function recoder_7k_best5() {
#     experiment_1=$1
#     batch_size=$2
#     max_token=$3
#     gpu=$4
#     bash call_scripts/train_nat.sh \
#             -e $experiment_1 --fp16 -g 2 \
#             --save-interval-updates 70000 --max-update 70000 --lm-start-step 75000 \
#             --lm-mask-rate 0.15 \
#             --max-tokens $max_token -b $batch_size -g $gpu
#     mkdir checkpoints/$experiment_1/top5_70000steps    
#     cp checkpoints/$experiment_1/checkpoint.best_bleu_*  checkpoints/$experiment_1/top5_70000steps

#     bash call_scripts/train_nat.sh \
#             -e $experiment_1 --fp16 -g 2 \
#             --save-interval-updates 70000 --max-update 100000 --lm-start-step 75000 \
#             --lm-mask-rate 0.15 \
#             --max-tokens $max_token -b $batch_size -g $gpu
# }




# #iwslt en-de
# recoder_7k_best5 2-2-1-3-B12-UD25B-lmx015 12288 3072 2

# #iwslt en-de
# recoder_7k_best5 2-2-2-3-B12-UD25B-lmx015 12288 3072 2


# #iwslt en-de
# recoder_7k_best5 2-2-3-3-B12-UR40B-lmx015 12288 2048 2





# #4
# bash call_scripts/train_nat.sh -e m-B-1-3-C12-UF20M-lm5 \
#                            --save-interval-updates 70000 --max-tokens 4096 \
#                            --has-eos --max-update 100000 --lm-start-step 75000 \
#                            --lm-iter-num 5 \
#                            -g 2 --fp16  




# function recoder_7k_best5() {
#     experiment_1=$1
#     batch_size=$2
#     max_token=$3
#     gpu=$4
#     bash call_scripts/train_nat.sh \
#             -e $experiment_1 --fp16 -g 2 \
#             --save-interval-updates 70000 --max-update 70000 --lm-start-step 75000 \
#             --lm-mask-rate 0.15 \
#             --max-tokens $max_token -b $batch_size -g $gpu
#     mkdir checkpoints/$experiment_1/top5_70000steps    
#     cp checkpoints/$experiment_1/checkpoint.best_bleu_*  checkpoints/$experiment_1/top5_70000steps

#     bash call_scripts/train_nat.sh \
#             -e $experiment_1 --fp16 -g 2 \
#             --save-interval-updates 70000 --max-update 100000 --lm-start-step 75000 \
#             --lm-mask-rate 0.15 \
#             --max-tokens $max_token -b $batch_size -g $gpu
# }


# #5
# #iwslt en-de
# recoder_7k_best5 2-2-1-3-B12-UD25M-lmx015 12288 3072 2

# #6
# recoder_7k_best5 m-B-1-3-C12-UF20M-lmx015 12288 6144 2


# #7
# recoder_7k_best5 m-B-3-3-C12-UF20M-lmx015 12288 6144 2


# #8
# bash call_scripts/train_nat.sh -e 2-2-1-3-B12-UD25M-eos \
#                                 --save-interval-updates 70000 --max-tokens 3072 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 -g 2 --fp16  




# bash call_scripts/train_nat.sh -e m-B-3-3-N-UF60M-Ltest \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 \
#                                 -g 1 --fp16   

# bash call_scripts/train_nat.sh -e m-B-3-3-N-UF70M-Ltest \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 \
#                                 -g 1 --fp16                                   


# bash call_scripts/train_nat.sh -e 2-2-1-1-H12-UF20M-lmx015-rate-pred \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 --lm-mask-rate 0.15 \
#                                 --arch ctcpmlm_rate_pred \
#                                 -g 1 --fp16   



# bash call_scripts/train_nat.sh -e m-B-3-1-N-UF60M-Ltest \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 \
#                                 -g 1 --fp16   
                                                     


# function pair_experiment_iwslt14() { 
#     relay_step=30000
#     LM_START_STEP=30000
#     MAX_TOKENS=768
#     GPU_NUM=1
#     BATCH_SIZE=12288
#     WARMUP_UPDATES=10000
#     MAX_UPDATE=50000
#     if [ -e checkpoints/$1/checkpoint_last.pt ]; then
#         echo "===========Loading $1 checkpoint_last step=============="
#         cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
#                   | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
#         echo "Currect step: $cur_last"
#     else
#         cur_last=0        
#     fi

#     if [ "$cur_last" -lt $relay_step ]; then    
#         bash call_scripts/train_nat.sh -e $1 \
#                                         --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
#                                         --lm-start-step $LM_START_STEP \
#                                         --task translation_ctcpmlm \
#                                         --arch nat_pretrained_model \
#                                         --criterion nat_ctc_loss \
#                                         --has-eos --max-update $relay_step \
#                                         --warmup-updates $WARMUP_UPDATES \
#                                         --hydra \
#                                         -g $GPU_NUM --fp16   
#     else
#         echo "$1 last step is ge $relay_step"
#     fi                                        
#     if [ -e checkpoints/$1/checkpoint_last.pt ]; then
#         echo "===========Loading $1 checkpoint_last step=============="
#         cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
#                   | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
#         echo "Currect step: $cur_last"
#     else
#         cur_last=0        
#     fi
#     if [ "$cur_last" -ge $relay_step ]; then
#         if [ -e checkpoints/$1/top5_$relay_step/checkpoint_last.pt ] && \
#         [ $(ls checkpoints/$1/top5_$relay_step/checkpoint.best_bleu_* 2>/dev/null \
#                 | grep -c "^checkpoints/$1/top5_$relay_step/checkpoint.best_bleu_.*") -eq 5 ]; then  
#             echo "$1 6 checkpoint in top5_$relay_step"
#         else
#             echo "save top 5 before $relay_step"
#             mkdir checkpoints/$1/top5_$relay_step
#             cp checkpoints/$1/checkpoint.best_bleu_* checkpoints/$1/top5_$relay_step
#             cp checkpoints/$1/checkpoint_last.pt checkpoints/$1/top5_$relay_step
#         fi
#         for experiment in $2 $3 $4; do
#             if [ -e checkpoints/$experiment/checkpoint_last.pt ] && \
#             [ $(ls checkpoints/$experiment/checkpoint.best_bleu_* 2>/dev/null | grep -c "^checkpoints/$experiment/checkpoint.best_bleu_.*") -eq 5 ]; then    
#                 echo "$experiment 6 checkpoint files exist"
#             else 
#                 mkdir checkpoints/$experiment/
#                 cp checkpoints/$1/top5_$relay_step/checkpoint.best_bleu_* checkpoints/$experiment/
#                 cp checkpoints/$1/top5_$relay_step/checkpoint_last.pt checkpoints/$experiment/     
#             fi     
#         done
#     fi
    
#     for experiment in $1 $2 $3 $4; do
#         bash call_scripts/train_nat.sh -e $experiment \
#                                         --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
#                                         --lm-start-step $LM_START_STEP \
#                                         --task translation_ctcpmlm \
#                                         --arch nat_pretrained_model \
#                                         --criterion nat_ctc_loss \
#                                         --has-eos --max-update $MAX_UPDATE \
#                                         --warmup-updates $WARMUP_UPDATES \
#                                         --hydra \
#                                         -g $GPU_NUM --fp16        
#     done                                                                                                                                                

# }
# pair_experiment_iwslt14 J-6-3-1-N-UR40M J-6-3-1-H12-UR40M
# pair_experiment J-6-3-1-N-UF30T
# pair_experiment I-6-3-1-N-UF30T
# pair_experiment K-6-3-1-N-UF30T

# source call_scripts/train/pair_experiment.sh
# pair_experiment_wmt14_3080x1 b-6-3-1-N-UF30T