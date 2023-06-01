source $HOME/.bashrc 
conda activate base

# bash call_scripts/train_nat.sh -e  a-2-1-1-H12-UR40M --fp16 -g 1 --save-interval-updates 10000 --max-update 100000 \
#                                                      --lm-start-step 75000 --max-tokens 2048 -b 65536 --no-atten-mask --has-eos


#bash call_scripts/train_nat.sh -e m-8-3-1-K12-UF20M-dp001 \
#                               --save-interval-updates 70000 --max-tokens 6144 \
#                               --has-eos --max-update 100000 --lm-start-step 75000 \
#                               --g 2 --fp16 --dropout 0.01  

#bash call_scripts/train_nat.sh -e m-8-3-3-K12-UF20M-dp001 \
#                               --save-interval-updates 70000 --max-tokens 6144 \
#                               --has-eos --max-update 100000 --lm-start-step 75000 \
#                               --g 2 --fp16 --dropout 0.01

#3
#bash call_scripts/train_nat.sh -e m-8-3-5-K12-UF20M-dp02 \
#                               --save-interval-updates 70000 --max-tokens 6144 \
#                               --has-eos --max-update 100000 --lm-start-step 75000 \
#                               --g 2 --fp16 --dropout 0.2


#4
#bash call_scripts/train_nat.sh -e m-8-1-5-K12-UF20M-dp02 \
#                               --save-interval-updates 70000 --max-tokens 6144 \
#                               --has-eos --max-update 100000 --lm-start-step 75000 \
#                               --g 2 --fp16 --dropout 0.2


#5
#bash call_scripts/train_nat.sh -e m-8-2-1-K12-UF20M-dp02 \
#                               --save-interval-updates 70000 --max-tokens 2048 \
#                               --has-eos --max-update 100000 --lm-start-step 75000 \
#                               --g 1 --fp16 --dropout 0.2


# #6
# bash call_scripts/train_nat.sh -e m-8-2-3-K12-UF20M-dp02 \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16 --dropout 0.2

# #7
# bash call_scripts/train_nat.sh -e m-8-2-5-K12-UF20M-dp02 \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16 --dropout 0.2


# #8
# bash call_scripts/train_nat.sh -e m-8-4-3-K12-UF20M-dp02 \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16 --dropout 0.2 

# #9
# bash call_scripts/train_nat.sh -e m-8-4-5-K12-UF20M-dp02 \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16 --dropout 0.2 




# function pair_experiment() { 
#     bash call_scripts/train_nat.sh -e $1 \
#                                     --save-interval-updates 70000 --max-tokens 2048 \
#                                     --lm-start-step 75000 \
#                                     --task translation_ctcpmlm \
#                                     --arch nat_pretrained_model \
#                                     --criterion nat_ctc_loss \
#                                     --has-eos --max-update 70000 \
#                                     --hydra \
#                                     -g 3 --fp16       

#     # for experiment in $2 $3 $4; do
#     #     mkdir checkpoints/$experiment/
#     #     cp checkpoints/$1/checkpoint.best_bleu_* checkpoints/$experiment/
#     #     cp checkpoints/$1/checkpoint_last.pt checkpoints/$experiment/
#     # done
    
#     for experiment in $1 $2 $3 $4; do
#         bash call_scripts/train_nat.sh -e $experiment \
#                                         --save-interval-updates 70000 --max-tokens 2048 \
#                                         --lm-start-step 75000 \
#                                         --task translation_ctcpmlm \
#                                         --arch nat_pretrained_model \
#                                         --criterion nat_ctc_loss \
#                                         --has-eos --max-update 100000 \
#                                         --hydra \
#                                         -g 3 --fp16        
#     done                                                                                                                                                

# }

# pair_experiment 2-2-3-1-H12-UF20M 2-2-3-1-N-UF20M 2-2-3-1-H7-UF20M 2-2-3-1-H4-UF20M





# function pair_experiment() { 
#     bash call_scripts/train_nat.sh -e $1 \
#                                     --save-interval-updates 70000 --max-tokens 2048 \
#                                     --lm-start-step 75000 \
#                                     --task translation_ctcpmlm \
#                                     --arch nat_pretrained_model \
#                                     --criterion nat_ctc_loss \
#                                     --has-eos --max-update 70000 \
#                                     --hydra \
#                                     -g 3 --fp16       


#         for experiment in $2 ; do
#             if [ -e checkpoints/$experiment/checkpoint_last.pt ] && \
#             [ $(ls checkpoints/$experiment/checkpoint.best_bleu_* 2>/dev/null | grep -c "^checkpoints/$experiment/checkpoint.best_bleu_.*") -eq 5 ]; then
#                 echo "All 6 checkpoint files exist"
#             else        
#                 mkdir checkpoints/$experiment/
#                 cp checkpoints/$1/checkpoint.best_bleu_* checkpoints/$experiment/
#                 cp checkpoints/$1/checkpoint_last.pt checkpoints/$experiment/
#             fi
#         done

#     for experiment in $1 $2; do
#         bash call_scripts/train_nat.sh -e $experiment \
#                                         --save-interval-updates 70000 --max-tokens 2048 \
#                                         --lm-start-step 75000 \
#                                         --task translation_ctcpmlm \
#                                         --arch nat_pretrained_model \
#                                         --criterion nat_ctc_loss \
#                                         --has-eos --max-update 100000 \
#                                         --hydra \
#                                         -g 3 --fp16        
#     done                                                                                                                                                

# }

# pair_experiment m-B-3-1-H12-UF20M m-B-3-1-N-UF20M 
# pair_experiment m-B-1-1-H12-UR20M m-B-1-1-N-UR20M





# function pair_experiment() { 
#     relay_step=70000
#     if [ -e checkpoints/$1/checkpoint_last.pt ]; then
#         echo "===========Loading $1 checkpoint_last step=============="
#         cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
#                   | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
#         echo "Currect step: $cur_last"
#     fi

#     if [ "$cur_last" -lt $relay_step ]; then    
#         bash call_scripts/train_nat.sh -e $1 \
#                                         --save-interval-updates $relay_step --max-tokens 1536 \
#                                         --lm-start-step 75000 \
#                                         --task translation_ctcpmlm \
#                                         --arch nat_pretrained_model \
#                                         --criterion nat_ctc_loss \
#                                         --has-eos --max-update $relay_step \
#                                         --hydra \
#                                         -g 2 --fp16   
#     else
#         echo "$1 last step is ge $relay_step"
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
#                                         --save-interval-updates $relay_step --max-tokens 1536 \
#                                         --lm-start-step 75000 \
#                                         --task translation_ctcpmlm \
#                                         --arch nat_pretrained_model \
#                                         --criterion nat_ctc_loss \
#                                         --has-eos --max-update 100000 \
#                                         --hydra \
#                                         -g 2 --fp16        
#     done                                                                                                                                                

# }
# # pair_experiment 2-2-1-1-H7-UF20T 2-2-1-1-N-UF20T
# # pair_experiment J-2-1-1-H7-UF20M J-2-1-1-N-UF20M J-2-1-1-H12-UF20M
# # pair_experiment J-2-1-1-H7-UF20T J-2-1-1-N-UF20T J-2-1-1-H12-UF20T
# # pair_experiment J-6-1-1-H7-UF20M J-6-1-1-N-UF20M J-6-1-1-H12-UF20M 
# pair_experiment J-6-1-1-H7-UF20T J-6-1-1-N-UF20T J-6-1-1-H12-UF20T 



# function pair_experiment() { 
#     relay_step=30000
#     LM_START_STEP=30000
#     max_tokens=1024
#     GPU_NUM=2
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
#                                         --save-interval-updates $relay_step --max-tokens $max_tokens \
#                                         --lm-start-step $LM_START_STEP \
#                                         --task translation_ctcpmlm \
#                                         --arch nat_pretrained_model \
#                                         --criterion nat_ctc_loss \
#                                         --has-eos --max-update $relay_step \
#                                         --lm-start-step $LM_START_STEP \
#                                         --hydra \
#                                         -g $GPU_NUM --fp16   
#     else
#         echo "$1 last step is ge $relay_step"
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
#                                         --save-interval-updates $relay_step --max-tokens $max_tokens \
#                                         --lm-start-step $LM_START_STEP \
#                                         --task translation_ctcpmlm \
#                                         --arch nat_pretrained_model \
#                                         --criterion nat_ctc_loss \
#                                         --has-eos --max-update $MAX_UPDATE \
#                                         --hydra \
#                                         -g $GPU_NUM --fp16        
#     done                                                                                                                                                

# }
# pair_experiment K-6-3-1-N-UF30T
# pair_experiment K-2-3-1-N-UF30T

# pair_experiment 2-2-1-1-H1-UF20M 2-2-1-1-H2-UF20M 2-2-1-1-H3-UF20M 2-2-1-1-H5-UF20M 
# pair_experiment 2-2-1-1-H1-UF20M 2-2-1-1-H6-UF20M 2-2-1-1-H8-UF20M 2-2-1-1-H9-UF20M
# pair_experiment 2-2-1-1-H1-UF20M 2-2-1-1-H10-UF20M 2-2-1-1-H11-UF20M 


source call_scripts/train/pair_experiment.sh
# pair_experiment_wmt14_1_2048_100k Z-6-3-1-N-UF30T
# pair_experiment_iwslt14_3080x1_768_50k J-2-3-1-N-UR40T J-2-3-1-H12-UR40T


