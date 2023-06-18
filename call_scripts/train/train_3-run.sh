source $HOME/.bashrc 
conda activate base
source call_scripts/train/pair_experiment.sh
# bash call_scripts/train_nat.sh -e K-2-1-1-H12-UR40M --fp16 --save-interval-updates 70000 --max-tokens 2048 --no-atten-mask
# bash call_scripts/train_nat.sh -e E-2-1-1-H12-UD25M --fp16 -g 1 --save-interval-updates 32500 --max-update 200000 --lm-start-step 130000 --max-tokens 3276 -b 65520 --no-atten-mask
# bash call_scripts/train_nat.sh -e L-5-1-1-N-UF30T-warmup_3k-table_12 -g 1 \
#     --max-update 30000 --save-interval-updates 10000 --max-tokens 2048 -b 65536 --fp16 \
#     --lm-start-step 50000 \
#     --dropout 0.1 \
#     --warmup-updates 3000 \
#     --no-atten-mask \

    
# bash call_scripts/generate_nat.sh -b 50 --data-subset test-valid --avg-ck-turnoff --no-atten-mask \
# -e L-5-1-1-N-UF30T-warmup_3k-table_12 \
# bash call_scripts/train_nat.sh -e I-2-1-1-H12-UR40M --fp16 -g 1 --save-interval-updates 70000 --max-tokens 2048 --no-atten-mask 

# bash call_scripts/wmt14_en_de/prepare_data/BiBert/generate-data-wmt.sh
# source $HOME/.bashrc 
# conda activate base
# bash call_scripts/train_nat.sh -e  R-6-1-1-N-UF30T --fp16 -g 1 --save-interval-updates 32500 --max-update 200000 --lm-start-step 130000 --max-tokens 2048 -b 65536 --no-atten-mask --has-eos
# Stop bash call_scripts/train_nat.sh -e  U-2-1-1-N-UF30T --fp16 -g 1 --save-interval-updates 32500 --max-update 200000 --lm-start-step 130000 --max-tokens 2048 -b 65536 --no-atten-mask --has-eos
# bash call_scripts/train_nat.sh -e  Z-2-1-1-N-UF30T --fp16 -g 2 --save-interval-updates 10000 --max-update 100000 \
#                                                     --lm-start-step 75000 --max-tokens 2048 -b 65536 --no-atten-mask --has-eos
# bash call_scripts/train_nat.sh -e  b-2-1-1-H12-UR40M --fp16 -g 2 --save-interval-updates 10000 --max-update 100000 \
#                                                     --lm-start-step 75000 --max-tokens 2048 -b 65536 --no-atten-mask --has-eos
# bash call_scripts/train_nat.sh -e  K-6-4-1-N-UF30T --fp16 -g 2 --save-interval-updates 70000 --max-update 100000 --lm-start-step 75000 --max-tokens 3072 -b 12288 --no-atten-mask 
# bash call_scripts/train_nat.sh -e  I-6-4-1-N-UF30T --fp16 -g 2 --save-interval-updates 70000 --max-update 100000 --lm-start-step 75000 --max-tokens 3072 -b 12288 --no-atten-mask
# bash call_scripts/train_nat.sh -e  I-2-1-1-H12-UR40M --fp16 -g 2 --save-interval-updates 70000 --max-update 100000 --lm-start-step 75000 --max-tokens 2048 -b 12288 --no-atten-mask
# bash call_scripts/train_nat.sh -e  P-2-1-1-H12-UD25M --fp16 -g 2 --save-interval-updates 70000 --max-update 100000 --lm-start-step 75000 --max-tokens 3072 -b 12288 --no-atten-mask
# bash call_scripts/train_nat.sh -e  1-5-4-1-H12-UF20T --fp16 -g 1 --save-interval-updates 70000 --max-update 100000 --lm-start-step 75000 --max-tokens 2048 -b 12288 --no-atten-mask 

# function pair_experiment() {
#     experiment_1=$1
#     experiment_2=$2
#     bash call_scripts/train_nat.sh \
#             -e  $experiment_1 --fp16 -g 2 --save-interval-updates 70000 --max-update 70000 \
#             --lm-start-step 75000 --max-tokens 1024 -b 12288 --no-atten-mask
#     mkdir checkpoints/$experiment_1/top5_70000steps    
#     cp checkpoints/$experiment_1/checkpoint.best_bleu_*  checkpoints/$experiment_1/top5_70000steps
#     mkdir checkpoints/$experiment_2/
#     cp checkpoints/$experiment_1/top5_70000steps/* checkpoints/$experiment_2/
#     cp checkpoints/$experiment_1/checkpoint_last.pt checkpoints/$experiment_2/
#     bash call_scripts/train_nat.sh \
#             -e  $experiment_1 --fp16 -g 2 --save-interval-updates 70000 --max-update 100000 \
#             --lm-start-step 75000 --max-tokens 1024 -b 12288 --no-atten-mask
# }

# pair_experiment 3-1-1-1-H12-UR40M 3-1-1-1-N-UR40M

# pair_experiment i-7-1-1-H12-UR40M i-7-1-1-N-UR40M

# bash call_scripts/generate_nat.sh -b 1 --data-subset test --ck-types top --avg-speed 1 --no-atten-mask \
# -e 3-1-1-1-H12-UR40M \
# -e i-1-1-1-H12-UR40M

# bash call_scripts/generate/generate_file-ship-speed-test.sh

# pair_experiment i-7-1-1-H12-UD25M i-7-1-1-N-UD25M


# bash call_scripts/train_nat.sh -e n-9-3-2-K12-UF20M  \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16


# bash call_scripts/train_nat.sh -e m-8-1-1-K12-UR40M-AutoModelForMaskedLM-randPos  \
#                                --save-interval-updates 70000 --max-tokens 3072 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16


# bash call_scripts/train_nat.sh -e m-8-4-1-K12-UR40M-AutoModelForMaskedLM-randPos  \
#                                --save-interval-updates 70000 --max-tokens 3072 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16


# bash call_scripts/train_nat.sh -e m-8-3-3-K12-UF20B \
#                                --save-interval-updates 70000 --max-tokens 4096 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16

# bash call_scripts/train_nat.sh -e m-8-4-3-K12-UF20B \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16

# bash call_scripts/train_nat.sh -e m-8-3-4-K12-UR40B \
#                                --save-interval-updates 70000 --max-tokens 3072 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16

# bash call_scripts/train_nat.sh -e m-8-2-4-K12-UR40B \
#                                --save-interval-updates 70000 --max-tokens 3072 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16


# bash call_scripts/train_nat.sh -e m-8-4-4-K12-UR40B \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16


# bash call_scripts/train_nat.sh -e m-8-3-5-K12-UR40B \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16

# bash call_scripts/train_nat.sh -e m-8-1-5-K12-UR40B \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16


# bash call_scripts/train_nat.sh -e m-8-2-5-K12-UR40B \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16

# bash call_scripts/train_nat.sh -e m-8-4-5-K12-UR40B \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16                               


# bash call_scripts/train_nat.sh -e m-8-3-5-K12-UR40B-dp001  \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16 --dropout 0.01


# #7
# bash call_scripts/train_nat.sh -e m-8-3-5-K12-UR40B-dp02 \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16 --dropout 0.2



# #8
# bash call_scripts/train_nat.sh -e m-8-1-5-K12-UR40B-dp02 \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16 --dropout 0.2



# #9
# bash call_scripts/train_nat.sh -e m-8-1-5-K12-UR40B-dp001 \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
                        #        --g 1 --fp16 --dropout 0.01


# #1
# bash call_scripts/train_nat.sh -e o-C-2-1-H12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16


# #2
# bash call_scripts/train_nat.sh -e o-C-2-1-K12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16

# #3
# bash call_scripts/train_nat.sh -e o-C-2-3-K12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16

# #4
# bash call_scripts/train_nat.sh -e o-C-2-4-K12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16  


# #5
# bash call_scripts/train_nat.sh -e o-C-4-3-K12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16

# #6
# bash call_scripts/train_nat.sh -e o-C-4-4-K12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16  



# #1
# bash call_scripts/train_nat.sh -e m-8-1-1-A12-UF20M-lm5 \
#                             --save-interval-updates 70000 --max-tokens 6144 \
#                             --has-eos --max-update 100000 --lm-start-step 75000 \
#                             --lm-iter-num 5 \
#                             -g 1 --fp16  


# #2
# bash call_scripts/train_nat.sh -e m-B-3-1-A12-UF20M-lm5 \
#                             --save-interval-updates 70000 --max-tokens 6144 \
#                             --has-eos --max-update 100000 --lm-start-step 75000 \
#                             --lm-iter-num 5 \
#                             -g 1 --fp16  

# #3
# bash call_scripts/train_nat.sh -e m-B-3-3-C12-UF20M-lm5 \
#                             --save-interval-updates 70000 --max-tokens 6144 \
#                             --has-eos --max-update 100000 --lm-start-step 75000 \
#                             --lm-iter-num 5 \
#                             -g 1 --fp16  


##4
#bash call_scripts/train_nat.sh -e m-B-1-3-C12-UF20M-lm5 \
#                            --save-interval-updates 70000 --max-tokens 6144 \
#                            --has-eos --max-update 100000 --lm-start-step 75000 \
#                            --lm-iter-num 5 \
#                            -g 2 --fp16  


#5
#bash call_scripts/train_nat.sh -e m-B-1-3-B12-UF20M-lmx015 \
#                                --save-interval-updates 70000 --max-tokens 4096 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --lm-mask-rate 0.15 \
#                                -g 2 --fp16 




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



# #1
# #iwslt en-de
# recoder_7k_best5 2-2-3-3-B12-UD25B-lmx015 12288 3072 2
# #2
# # wmt14 en-de
# recoder_7k_best5 Y-2-3-3-B12-UD25B-lmx015 65536 3072 2


# #3
# #iwslt en-de
# recoder_7k_best5 2-2-3-3-B12-UD25M-lmx015 12288 3072 2


#4
#iwslt en-de
# recoder_7k_best5 2-2-2-3-B12-UD25M-lmx015 12288 3072 1





# #5
# bash call_scripts/train_nat.sh -e p-B-3-3-B12-UF20M-lmx015-cont \
#                                 --save-interval-updates 70000 --max-tokens 6144 \
#                                 --has-eos --max-update 200000 --lm-start-step 150000 \
#                                 --lm-mask-rate 0.15 --reset-meters \
#                                 -g 1 --fp16    



# bash call_scripts/train_nat.sh -e m-B-3-3-N-UF40M-Ltest \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 \
#                                 -g 2 --fp16       

# bash call_scripts/train_nat.sh -e m-B-3-3-N-UF50M-Ltest \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 \
#                                 -g 2 --fp16  


# bash call_scripts/train_nat.sh -e m-B-1-1-N-UF20M   \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 \
#                                 -g 2 --fp16  
                                 

# bash call_scripts/train_nat.sh -e m-B-1-3-H12-UF20M   \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 \
#                                 -g 2 --fp16  


# bash call_scripts/train_nat.sh -e m-B-1-1-N-UF20M-rate-test \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 --debug \
#                                 -g 1 --fp16 




# #8
# bash call_scripts/train_nat.sh -e m-B-3-3-B12-UR20M-lmx015-rate-pred \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 --lm-mask-rate 0.15 \
#                                 --arch ctcpmlm_rate_pred \
#                                 -g 2 --fp16   


# bash call_scripts/train_nat.sh -e m-B-1-3-B12-UR20M-lmx015-rate-pred \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 --lm-mask-rate 0.15 \
#                                 --arch ctcpmlm_rate_pred \
#                                 -g 2 --fp16   


# bash call_scripts/train_nat.sh -e m-B-3-1-N-UF40M-Ltest \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 \
#                                 -g 2 --fp16       

# bash call_scripts/train_nat.sh -e m-B-3-1-N-UF50M-Ltest \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 \
#                                 -g 2 --fp16  



# bash call_scripts/train_nat.sh -e m-B-1-1-N-UR20M-Ltest \
#                                 --save-interval-updates 70000 --max-tokens 4096 \
#                                 --task translation_ctcpmlm \
#                                 --arch nat_pretrained_model \
#                                 --criterion nat_ctc_loss \
#                                 --has-eos --max-update 100000 \
#                                 -g 3 --fp16       
                            

# bash call_scripts/train_nat.sh -e m-B-1-1-N-UR40M-Ltest \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --task translation_ctcpmlm \
#                                 --arch nat_pretrained_model \
#                                 --criterion nat_ctc_loss \
#                                 --has-eos --max-update 100000 \
#                                 -g 3 --fp16 




# bash call_scripts/train_nat.sh -e m-B-1-1-N-UR20M-rate_predict_divTGT \
#                                 --save-interval-updates 70000 --max-tokens 4096 \
#                                 --arch ctcpmlm_rate_predictor \
#                                 --task transaltion_ctcpmlm_rate \
#                                 --criterion nat_ctc_pred_rate_loss \
#                                 --hydra \
#                                 --debug \
#                                 -g 1 --fp16    


# bash call_scripts/train_nat.sh -e m-B-1-1-N-UR20M-rate_predict_divTGT-NEW \
#                                 --save-interval-updates 70000 --max-tokens 4096 \
#                                 --arch ctcpmlm_rate_predictor \
#                                 --task transaltion_ctcpmlm_rate \
#                                 --criterion nat_ctc_pred_rate_loss \
#                                 --hydra \
#                                 --debug \
#                                 -g 1 --fp16   




# bash call_scripts/train_nat.sh -e m-B-1-1-N-UR20M-rate_select-divTGT-NEW-2 \
#                                 --save-interval-updates 70000 --max-tokens 3072 \
#                                 --arch ctcpmlm_rate_selection \
#                                 --task translation_ctcpmlm \
#                                 --criterion nat_ctc_sel_rate_loss \
#                                 --has-eos --max-update 100000 \
#                                 --hydra \
#                                 --debug \
#                                 -g 1 --fp16


# bash call_scripts/train_nat.sh -e m-B-1-1-N-UR20M-rate_select-divTGT-NEW-3 \
#                                 --save-interval-updates 70000 --max-tokens 3072 \
#                                 --arch ctcpmlm_rate_selection \
#                                 --task translation_ctcpmlm \
#                                 --criterion nat_ctc_sel_rate_loss \
#                                 --has-eos --max-update 100000 \
#                                 --hydra \
#                                 -g 1 --fp16


# bash call_scripts/train_nat.sh -e m-B-1-1-N-UR20M-rate_avg \
#                                 --save-interval-updates 70000 --max-tokens 3072 \
#                                 --arch ctcpmlm_rate_selection \
#                                 --task translation_ctcpmlm \
#                                 --criterion nat_ctc_avg_rate_loss \
#                                 --has-eos --max-update 100000 \
#                                 --hydra \
#                                 -g 1 --fp16




# function pair_experiment_1() { 
#     bash call_scripts/train_nat.sh -e $1 \
#                                     --save-interval-updates 70000 --max-tokens 3072 \
#                                     --lm-start-step 75000 \
#                                     --task translation_ctcpmlm \
#                                     --arch nat_pretrained_model \
#                                     --criterion nat_ctc_loss \
#                                     --has-eos --max-update 70000 \
#                                     --hydra \
#                                     -g 1 --fp16       


#         for experiment in $2 $3 $4 ; do
#             if [ -e checkpoints/$experiment/checkpoint_last.pt ] && \
#             [ $(ls checkpoints/$experiment/checkpoint.best_bleu_* 2>/dev/null | grep -c "^checkpoints/$experiment/checkpoint.best_bleu_.*") -eq 5 ]; then
#                 echo "All 6 checkpoint files exist"
#             else        
#                 mkdir checkpoints/$experiment/
#                 cp checkpoints/$1/checkpoint.best_bleu_* checkpoints/$experiment/
#                 cp checkpoints/$1/checkpoint_last.pt checkpoints/$experiment/
#             fi
#         done

#     for experiment in $1 $2 $3 $4 ; do
#         bash call_scripts/train_nat.sh -e $experiment \
#                                         --save-interval-updates 70000 --max-tokens 3072 \
#                                         --lm-start-step 75000 \
#                                         --task translation_ctcpmlm \
#                                         --arch nat_pretrained_model \
#                                         --criterion nat_ctc_loss \
#                                         --has-eos --max-update 100000 \
#                                         --hydra \
#                                         -g 1 --fp16        
#     done                                                                                                                                                

# }

# pair_experiment_1 K-2-1-1-H12-UR40M K-2-1-1-N-UR40M


# function pair_experiment_1() { 
#     bash call_scripts/train_nat.sh -e $1 \
#                                     --save-interval-updates 70000 --max-tokens 3072 \
#                                     --lm-start-step 75000 \
#                                     --task translation_ctcpmlm \
#                                     --arch nat_pretrained_model \
#                                     --criterion nat_ctc_loss \
#                                     --has-eos --max-update 70000 \
#                                     --hydra \
#                                     -g 2 --fp16       


#         for experiment in $2 $3 $4 ; do
#             if [ -e checkpoints/$experiment/checkpoint_last.pt ] && \
#             [ $(ls checkpoints/$experiment/checkpoint.best_bleu_* 2>/dev/null | grep -c "^checkpoints/$experiment/checkpoint.best_bleu_.*") -eq 5 ]; then
#                 echo "All 6 checkpoint files exist"
#             else        
#                 mkdir checkpoints/$experiment/
#                 cp checkpoints/$1/checkpoint.best_bleu_* checkpoints/$experiment/
#                 cp checkpoints/$1/checkpoint_last.pt checkpoints/$experiment/
#             fi
#         done

#     for experiment in $1 $2 $3 $4 ; do
#         bash call_scripts/train_nat.sh -e $experiment \
#                                         --save-interval-updates 70000 --max-tokens 3072 \
#                                         --lm-start-step 75000 \
#                                         --task translation_ctcpmlm \
#                                         --arch nat_pretrained_model \
#                                         --criterion nat_ctc_loss \
#                                         --has-eos --max-update 100000 \
#                                         --hydra \
#                                         -g 2 --fp16        
#     done                                                                                                                                                

# }

# pair_experiment_1 2-6-1-1-H7-UF20M 2-6-1-1-N-UF20M 2-6-1-1-H12-UF20M 

# bash call_scripts/train_nat.sh -e m-B-1-1-N-UR20M-rate_predict \
#                                 --save-interval-updates 70000 --max-tokens 1536 \
#                                 --arch ctcpmlm_rate_predictor \
#                                 --task transaltion_ctcpmlm_rate \
#                                 --criterion nat_ctc_pred_rate_loss \
#                                 --hydra \
#                                 -g 2 --fp16  

# function pair_experiment() { 
#     relay_step=70000
#     max_tokens=4096
#     GPU_NUM=2
#     if [ -e checkpoints/$1/checkpoint_last.pt ]; then
#         echo "===========Loading $1 checkpoint_last step=============="
#         cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
#                   | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
#         echo "Currect step: $cur_last"
#     fi

#     if [ "$cur_last" -lt $relay_step ]; then    
#         bash call_scripts/train_nat.sh -e $1 \
#                                         --save-interval-updates $relay_step --max-tokens $max_tokens \
#                                         --lm-start-step 75000 \
#                                         --task translation_ctcpmlm \
#                                         --arch nat_pretrained_model \
#                                         --criterion nat_ctc_loss \
#                                         --has-eos --max-update $relay_step \
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
#                                         --lm-start-step 75000 \
#                                         --task translation_ctcpmlm \
#                                         --arch nat_pretrained_model \
#                                         --criterion nat_ctc_loss \
#                                         --has-eos --max-update 100000 \
#                                         --hydra \
#                                         -g $GPU_NUM --fp16        
#     done                                                                                                                                                

# }


# pair_experiment 2-2-1-1-H12-UF20T 2-2-1-1-H4-UF20T 


# function pair_experiment() { 
#     relay_step=30000
#     LM_START_STEP=30000
#     max_tokens=2048
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
#     cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
#                 | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
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
# pair_experiment K-2-3-1-N-UR40M K-2-3-1-H12-UR40M
# pair_experiment K-2-3-1-N-UR40T K-2-3-1-H12-UR40T
# pair_experiment 2-6-3-1-N-UR40T 2-6-3-1-H12-UR40T


# function pair_experiment_wmt14() { 
#     relay_step=70000
#     LM_START_STEP=75000
#     MAX_TOKENS=2048
#     GPU_NUM=2
#     BATCH_SIZE=65536
#     WARMUP_UPDATES=10000
#     MAX_UPDATE=100000
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
#                                         -b $BATCH_SIZE \
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
#                                         -b $BATCH_SIZE \
#                                         --hydra \
#                                         -g $GPU_NUM --fp16        
#     done                                                                                                                                                

# }
# pair_experiment_wmt14 Z-2-3-1-N-UF30T


# pair_experiment_wmt14_2_2048_100k Z-6-3-1-N-UF30T

# pair_experiment_iwslt14_2_2048_50k J-2-3-1-N-UR40T J-2-3-1-H12-UR40T
# pair_experiment_iwslt14_2_2048_50k 1-1-3-1-H12-UR20M
# pair_experiment_iwslt14_2_2048_50k 1-1-3-1-H12-UR22M
# pair_experiment_iwslt14_2_2048_50k 1-1-3-1-H12-UR25M
# pair_experiment_iwslt14_2_2048_50k 1-1-3-1-H12-UR30M JCX
# pair_experiment_iwslt14_2_1024_50k 1-1-3-1-H12-UR33M JCX
# pair_experiment_iwslt14_1_1024_50k 2-2-3-1-H1-UR40M 2-2-3-1-H11-UR40M
# pair_experiment_iwslt14_1_1024_50k 1-1-3-1-H12-UR20M


# pair_experiment 2-2-3-1-H1-UF20T 2-2-3-1-H2-UF20T 2-2-3-1-H3-UF20T 2-2-3-1-H5-UF20T 
# pair_experiment 2-2-3-1-H1-UF20T 2-2-3-1-H6-UF20T 2-2-3-1-H8-UF20T 2-2-3-1-H9-UF20T
# pair_experiment 2-2-3-1-H1-UF20T 2-2-3-1-H10-UF20T 2-2-3-1-H11-UF20T 




# source call_scripts/train/pair_experiment.sh
# experiment=2-2-3-1-N-UR20M
# pair_experiment_iwslt14_1_4096_100k $experiment



