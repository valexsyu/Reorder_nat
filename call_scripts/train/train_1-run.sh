source $HOME/.bashrc 
conda activate base
source call_scripts/train/pair_experiment.sh

# bash call_scripts/train_nat.sh -e 2-6-1-1-N-UF30T --fp16 --save-interval-updates 70000 --max-tokens 2048 --no-atten-mask
# bash call_scripts/train_nat.sh -e K-2-1-1-H12-UD25M --fp16 --save-interval-updates 70000 --max-tokens 2048 --no-atten-mask
# bash call_scripts/train_nat.sh -e F-2-1-1-H12-UD25M --fp16 -g 1 --save-interval-updates 32500 --max-update 200000 --lm-start-step 130000 --max-tokens 3276 -b 65520 --no-atten-mask
# bash call_scripts/train_nat.sh -e J-6-1-1-N-UF30T --fp16 --save-interval-updates 70000 --max-tokens 2048 --no-atten-mask
# bash call_scripts/train_nat.sh -e 2-2-1-1-H4-UR40M --fp16 --save-interval-updates 70000 --max-tokens 2048 --no-atten-mask
#bash call_scripts/train_nat.sh -e P-2-1-1-H12-UD25M --fp16 -g 1 --save-interval-updates 70000 --max-tokens 3072 --no-atten-mask
# bash call_scripts/train_nat.sh -e J-2-1-1-H12-UR40M --fp16 -g 1 --save-interval-updates 70000 --max-tokens 2048 --no-atten-mask 
# bash call_scripts/train_nat.sh -e  S-6-1-1-N-UF30T --fp16 -g 2 --save-interval-updates 32500 --max-update 200000 --lm-start-step 130000 --max-tokens 2048 -b 65536 --no-atten-mask --has-eos
#  Not done bash call_scripts/train_nat.sh -e  U-2-1-1-N-UR40T --fp16 -g 2 --save-interval-updates 10000 --max-update 100000 --lm-start-step 75000 --max-tokens 2048 -b 65536 --no-atten-mask --has-eos
# bash call_scripts/train_nat.sh -e  Z-2-1-1-N-UR40T --fp16 -g 2 --save-interval-updates 10000 --max-update 100000 --lm-start-step 75000 --max-tokens 2048 -b 65536 --no-atten-mask --has-eos
# bash call_scripts/train_nat.sh -e  a-6-1-1-N-UF30T --fp16 -g 1 --save-interval-updates 10000 --max-update 100000 --lm-start-step 75000 --max-tokens 4096 -b 65536 --no-atten-mask --has-eos
# bash call_scripts/train_nat.sh -e  2-6-4-1-N-UF30T --fp16 -g 1 --save-interval-updates 70000 --max-update 100000 --lm-start-step 75000 --max-tokens 4096 -b 12288 --no-atten-mask 
# bash call_scripts/train_nat.sh -e  J-6-4-1-N-UF30T --fp16 -g 1 --save-interval-updates 70000 --max-update 100000 --lm-start-step 75000 --max-tokens 4096 -b 12288 --no-atten-mask 
# bash call_scripts/train_nat.sh -e  4-2-1-1-H12-UD25M --fp16 -g 1 --save-interval-updates 70000 --max-update 100000 --lm-start-step 75000 --max-tokens 3072 -b 12288 --no-atten-mask 
# bash call_scripts/train_nat.sh -e  3-5-4-1-H12-UF20T --fp16 -g 1 --save-interval-updates 70000 --max-update 100000 --lm-start-step 75000 --max-tokens 3072 -b 12288 --no-atten-mask 


# function pair_experiment() {
#     experiment_1=$1
#     experiment_2=$2
#     bash call_scripts/train_nat.sh \
#             -e $experiment_1 --fp16 -g 2 \
#             --save-interval-updates 70000 --max-update 70000 --lm-start-step 75000 \
#             --max-tokens 2048 -b 65536 --no-atten-mask 
#     mkdir checkpoints/$experiment_1/top5_70000steps    
#     cp checkpoints/$experiment_1/checkpoint.best_bleu_*  checkpoints/$experiment_1/top5_70000steps
#     mkdir checkpoints/$experiment_2/
#     cp checkpoints/$experiment_1/top5_70000steps/* checkpoints/$experiment_2/
#     cp checkpoints/$experiment_1/checkpoint_last.pt checkpoints/$experiment_2/
#     bash call_scripts/train_nat.sh \
#             -e $experiment_1 --fp16 -g 2 \
#             --save-interval-updates 70000 --max-update 100000 --lm-start-step 75000 \
#             --max-tokens 2048 -b 65536 --no-atten-mask 
# }

# pair_experiment e-7-3-1-H12-UD25M e-7-3-1-N-UD25M

# bash call_scripts/train_nat.sh -e 1-1-1-1-H12-UF20M-new \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                -g 1 --fp16




# bash call_scripts/train_nat.sh -e m-8-3-1-K12-UF20M-AutoModelForMaskedLM-randPos  \
#                                --save-interval-updates 70000 --max-tokens 4096 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16


# bash call_scripts/train_nat.sh -e m-8-1-1-K12-UF20M-AutoModelForMaskedLM-randPos  \
#                                --save-interval-updates 70000 --max-tokens 4096 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16


# bash call_scripts/train_nat.sh -e m-8-3-1-K12-UR40M-AutoModelForMaskedLM-randPos  \
#                                --save-interval-updates 70000 --max-tokens 3072 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16



# bash call_scripts/train_nat.sh -e m-B-1-1-K12-UF20M-AutoModelForMaskedLM-randPos  \
#                                --save-interval-updates 70000 --max-tokens 4096 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16


# bash call_scripts/train_nat.sh -e m-8-1-3-K12-UF20B \
#                                --save-interval-updates 70000 --max-tokens 4096 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16


# bash call_scripts/train_nat.sh -e m-8-1-4-K12-UF20B \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16

# bash call_scripts/train_nat.sh -e m-8-3-4-K12-UF20B \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16                               


# bash call_scripts/train_nat.sh -e m-8-4-4-K12-UF20B \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16     

# bash call_scripts/train_nat.sh -e m-8-2-1-K12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16 

# bash call_scripts/train_nat.sh -e m-8-1-5-K12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16

# bash call_scripts/train_nat.sh -e m-8-1-5-K12-UF20B \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16


# bash call_scripts/train_nat.sh -e m-8-2-5-K12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16

# bash call_scripts/train_nat.sh -e m-8-2-5-K12-UF20B \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16



# bash call_scripts/train_nat.sh -e m-8-1-4-K12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16

# bash call_scripts/train_nat.sh -e m-8-2-4-K12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16

# bash call_scripts/train_nat.sh -e m-8-3-4-K12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16

# bash call_scripts/train_nat.sh -e m-8-4-4-K12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16 


# #11
# bash call_scripts/train_nat.sh -e m-8-1-3-K12-UF20M-dp001 \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16 --dropout 0.01

# #12
# bash call_scripts/train_nat.sh -e m-8-1-1-H12-UF20M-dp02 \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16 --dropout 0.2

# #13
# bash call_scripts/train_nat.sh -e m-8-2-1-H12-UF20M-dp02 \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16 --dropout 0.2


# #14
# bash call_scripts/train_nat.sh -e m-8-1-1-K12-UF20M-dp02 \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16 --dropout 0.2


# #15
# bash call_scripts/train_nat.sh -e m-8-1-3-K12-UF20M-dp02 \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16 --dropout 0.2

# #16
# bash call_scripts/train_nat.sh -e m-8-1-4-K12-UF20M-dp001 \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16 --dropout 0.01




# #1
# bash call_scripts/train_nat.sh -e o-C-1-1-H12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16


# #2
# bash call_scripts/train_nat.sh -e o-C-1-1-K12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16

# #3
# bash call_scripts/train_nat.sh -e o-C-1-3-K12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16

# #4
# bash call_scripts/train_nat.sh -e o-C-1-4-K12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16        


# #5
# bash call_scripts/train_nat.sh -e o-C-1-1-H12-UF20B \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16        

# #6
# bash call_scripts/train_nat.sh -e o-C-1-4-K12-UF20B \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16        

# #7
# bash call_scripts/train_nat.sh -e o-C-3-1-H12-UF20B \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16  


# #8
# bash call_scripts/train_nat.sh -e o-C-3-4-K12-UF20B \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16  


#  #1
#  bash call_scripts/train_nat.sh -e m-8-1-1-A12-UF20M \
#                                 --save-interval-updates 70000 --max-tokens 6144 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 -g 2 --fp16 



#  #2
#  bash call_scripts/train_nat.sh -e m-8-3-1-A12-UF20M \
#                                 --save-interval-updates 70000 --max-tokens 6144 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 -g 2 --fp16 



# #3
# bash call_scripts/train_nat.sh -e m-B-1-1-B12-UF20M-lm5 \
#                             --save-interval-updates 70000 --max-tokens 3072 \
#                             --has-eos --max-update 100000 --lm-start-step 75000 \
#                             --lm-iter-num 5 --watch-lm-loss \
#                             -g 2 --fp16 


# #4
# bash call_scripts/train_nat.sh -e m-B-1-3-B12-UF20M-lm5 \
#                             --save-interval-updates 70000 --max-tokens 3072 \
#                             --has-eos --max-update 100000 --lm-start-step 75000 \
#                             --lm-iter-num 5 \
#                             -g 2 --fp16 



# #5
# bash call_scripts/train_nat.sh -e m-B-1-1-C12-UF20M-lm5x015 \
#                                 --save-interval-updates 70000 --max-tokens 3072 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 --lm-iter-num 5 --lm-mask-rate 0.15 \
#                                 -g 2 --fp16 



# #  #6
# bash call_scripts/train_nat.sh -e m-B-1-1-A12-UF20M-lmx015 \
#                                 --save-interval-updates 70000 --max-tokens 6144 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 --lm-mask-rate 0.15 \
#                                 -g 2 --fp16 

# #  #7
# bash call_scripts/train_nat.sh -e m-B-1-1-A12-UF20M-lm5x015 \
#                                 --save-interval-updates 70000 --max-tokens 6144 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 --lm-iter-num 5 --lm-mask-rate 0.15 \
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



# #1
# #iwslt en-de 
# recoder_7k_best5 K-2-3-3-B12-UD25B-lmx015 12288 3072 2
# #2
# #wmt14 en-de
# recoder_7k_best5 Z-2-3-3-B12-UD25B-lmx015 65536 3072 2

# #3
# #iwslt en-de 
# recoder_7k_best5 K-2-3-3-B12-UD25M-lmx015 12288 3072 2

#4
#iwslt en-de
# recoder_7k_best5 2-2-3-3-B12-UR40M-lmx015 12288 3072 1


# #5
# bash call_scripts/train_nat.sh -e p-B-3-3-B12-UF20M-lmx015-optini \
#                                 --save-interval-updates 70000 --max-tokens 6144 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 --lm-mask-rate 0.15 \
#                                 -g 2 --fp16    
# --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer \


# bash call_scripts/train_nat.sh -e m-B-3-3-N-UF20M-Ltest \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 \
#                                 -g 2 --fp16    

# bash call_scripts/train_nat.sh -e m-B-3-3-N-UF30M-Ltest \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 \
#                                 -g 1 --fp16    


# bash call_scripts/train_nat.sh -e 1-1-1-1-H12-UF20M-eos \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 \
#                                 -g 2 --fp16  


# function pair_experiment() {
#     experiment_1=$1
#     experiment_2=$2
#     gpu=$3
#     bash call_scripts/train_nat.sh \
#             -e $experiment_1  \
#             --save-interval-updates 70000 --max-update 70000 --lm-start-step 75000 \
#             --max-tokens 2048 \
#             --arch ctcpmlm_rate_pred \
#             --fp16 -g $gpu
#     mkdir checkpoints/$experiment_1/top5_70000steps    
#     cp checkpoints/$experiment_1/checkpoint.best_bleu_*  checkpoints/$experiment_1/top5_70000steps
#     mkdir checkpoints/$experiment_2/
#     cp checkpoints/$experiment_1/top5_70000steps/* checkpoints/$experiment_2/
#     cp checkpoints/$experiment_1/checkpoint_last.pt checkpoints/$experiment_2/
#     bash call_scripts/train_nat.sh \
#             -e $experiment_1  \
#             --save-interval-updates 70000 --max-update 100000 --lm-start-step 75000 \
#             --max-tokens 2048 \
#             --arch ctcpmlm_rate_pred \
#             --fp16 -g $gpu
#     bash call_scripts/train_nat.sh \
#             -e $experiment_2  \
#             --save-interval-updates 70000 --max-update 100000 --lm-start-step 75000 \
#             --max-tokens 2048 \
#             --arch ctcpmlm_rate_pred \
#             --fp16 -g $gpu           
# }

# pair_experiment m-B-1-1-H12-UR20M-rate-pred m-B-1-1-N-UR20M-rate-pred 2
# pair_experiment m-B-3-1-H12-UR20M-rate-pred m-B-3-1-N-UR20M-rate-pred 2

 


# bash call_scripts/train_nat.sh -e m-B-3-1-N-UF20M-Ltest \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 \
#                                 -g 2 --fp16    

# bash call_scripts/train_nat.sh -e m-B-3-1-N-UF30M-Ltest \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 \
#                                 -g 2 --fp16  


                                

# bash call_scripts/train_nat.sh -e m-B-3-1-N-UF20M-Ltest-amp \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 \
#                                 -g 2 --amp


# bash call_scripts/train_nat.sh -e m-B-3-1-N-UR20M-rate_sel-5k-rate_2_3_4 \
#                                 --save-interval-updates 70000 --max-tokens 3072 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 --arch ctcpmlm_rate_selection \
#                                 --criterion nat_ctc_sel_rate_loss \
#                                 --lmax-only-step 5000 \
#                                 -g 2 --fp16 


# fairseq-hydra-train -r --config-dir checkpoints/m-B-1-1-N-UR20M-lmx015-test-777/  --config-name test_yaml.yaml 


# bash call_scripts/train_nat.sh -e m-B-3-1-N-UR20M-rate_sel-5k-rate_2_3_4 \
#                                 --save-interval-updates 70000 --max-tokens 1536 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 --arch ctcpmlm_rate_selection \
#                                 --criterion nat_ctc_sel_rate_loss \
#                                 --lmax-only-step 5000 \
#                                 -g 2 --fp16 


# fairseq-hydra-train -r --config-dir checkpoints/m-B-1-1-N-UR20M-rate_pred/  --config-name m-B-1-1-N-UR20M.yaml


# bash call_scripts/train_nat.sh -e m-B-1-1-N-UR20M-rate_pred \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --arch ctcpmlm_rate_predictor \
#                                 --task transaltion_ctcpmlm_rate \
#                                 --criterion nat_ctc_pred_rate_loss \
#                                 --hydra \
#                                 -g 2 --fp16


    


# bash call_scripts/train_nat.sh -e m-B-1-1-N-UR20M-rate_predict \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --arch ctcpmlm_rate_predictor \
#                                 --task transaltion_ctcpmlm_rate \
#                                 --criterion nat_ctc_pred_rate_loss \
#                                 --hydra \
#                                 -g 2 --fp16    
                            


# bash call_scripts/train_nat.sh -e m-B-1-1-N-UR20M-rate_select-divTGT \
#                                 --save-interval-updates 70000 --max-tokens 1536 \
#                                 --arch ctcpmlm_rate_selection \
#                                 --task translation_ctcpmlm \
#                                 --criterion nat_ctc_sel_rate_loss \
#                                 --has-eos --max-update 100000 \
#                                 --hydra \
#                                 --debug \
#                                 -g 2 --fp16


# bash call_scripts/train_nat.sh -e m-B-1-1-N-UR20M-rate_select-divTGT-NEW \
#                                 --save-interval-updates 70000 --max-tokens 1536 \
#                                 --arch ctcpmlm_rate_selection \
#                                 --task translation_ctcpmlm \
#                                 --criterion nat_ctc_sel_rate_loss \
#                                 --has-eos --max-update 100000 \
#                                 --hydra \
#                                 --debug \
#                                 -g 2 --fp16



# function pair_experiment() { 
#     bash call_scripts/train_nat.sh -e $1 \
#                                     --save-interval-updates 70000 --max-tokens 3072 \
#                                     --lm-start-step 75000 \
#                                     --task translation_ctcpmlm \
#                                     --arch nat_pretrained_model \
#                                     --criterion nat_ctc_loss \
#                                     --has-eos --max-update 70000 \
#                                     --hydra \
#                                     -g 2 --fp16   

#     for experiment in $2 $3 $4; do
#         if [ -e checkpoints/$experiment/checkpoint_last.pt ] && \
#         [ $(ls checkpoints/$experiment/checkpoint.best_bleu_* 2>/dev/null | grep -c "^checkpoints/$experiment/checkpoint.best_bleu_.*") -eq 5 ]; then    
#             echo "All 6 checkpoint files exist"
#         else 
#             mkdir checkpoints/$experiment/
#             cp checkpoints/$1/checkpoint.best_bleu_* checkpoints/$experiment/
#             cp checkpoints/$1/checkpoint_last.pt checkpoints/$experiment/     
#         fi     
#     done
    
#     for experiment in $1 $2 $3 $4; do
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

# pair_experiment 2-2-1-1-H12-UF20M 2-2-1-1-N-UF20M 2-2-1-1-H7-UF20M 2-2-1-1-H4-UF20M

# bash call_scripts/train_nat.sh -e 2-2-3-1-N-UR20M-rate_select-divTGT-NEW-3 \
#                                 --save-interval-updates 70000 --max-tokens 1536 \
#                                 --arch ctcpmlm_rate_selection \
#                                 --task translation_ctcpmlm \
#                                 --criterion nat_ctc_sel_rate_loss \
#                                 --has-eos --max-update 100000 \
#                                 --hydra \
#                                 -g 2 --fp16


# bash call_scripts/train_nat.sh -e m-B-1-1-N-UR20M-rate_predict_divTGT-NEW-detach-correct \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --arch ctcpmlm_rate_predictor \
#                                 --task transaltion_ctcpmlm_rate \
#                                 --criterion nat_ctc_pred_rate_loss \
#                                 --hydra \
#                                 -g 2 --fp16   


# function pair_experiment() { 
#     relay_step=70000
#     max_tokens=2048
#     GPU_NUM=1
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
# pair_experiment 2-6-1-1-H7-UF20T 2-6-1-1-N-UF20T 2-6-1-1-H12-UF20T 


# pair_experiment 2-2-3-1-H1-UF20M 2-2-3-1-H2-UF20M 2-2-3-1-H3-UF20M 2-2-3-1-H5-UF20M 

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
#             | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')

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
# pair_experiment 2-6-3-1-N-UF30T
# pair_experiment 2-2-3-1-N-UF30T
# pair_experiment J-6-3-1-N-UR40T J-6-3-1-H12-UR40T




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
# pair_experiment_wmt14 Z-2-3-1-N-UR40T

# pair_experiment_iwslt14_2_2048_50k m-B-3-1-H12-UR25M-50k
# pair_experiment_iwslt14_2_2048_50k m-B-3-1-H12-UR30M-50k
# pair_experiment_iwslt14_2_2048_50k m-B-3-1-H12-UR20M-50k
# pair_experiment_iwslt14_2_2048_50k m-B-3-1-H12-UR40M-50k
# pair_experiment_iwslt14_2_2048_50k m-B-3-1-H12-UR35M-50k
# pair_experiment_iwslt14_2_2048_50k m-B-3-1-H12-UR45M-50k

# pair_experiment 2-2-3-1-H1-UF20M 2-2-3-1-H6-UF20M 2-2-3-1-H8-UF20M 2-2-3-1-H9-UF20M
# pair_experiment 2-2-3-1-H1-UF20M 2-2-3-1-H10-UF20M 2-2-3-1-H11-UF20M 

# pair_experiment_iwslt14_2_2048_50k 2-2-3-1-H1-UR40M 2-2-3-1-H2-UR40M 2-2-3-1-H3-UR40M 2-2-3-1-H4-UR40M \
#                                    2-2-3-1-H5-UR40M 2-2-3-1-H6-UR40M 2-2-3-1-H7-UR40M 2-2-3-1-H8-UR40M \
#                                    2-2-3-1-H9-UR40M 2-2-3-1-H10-UR40M 2-2-3-1-H11-UR40M


# pair_experiment_iwslt14_2_2048_50k 2-2-3-1-H1-UR40M 2-2-3-1-H5-UR40M 2-2-3-1-H6-UR40M 2-2-3-1-H7-UR40M
# pair_experiment_iwslt14_2_2048_50k 2-2-3-1-H1-UR40M 2-2-3-1-H8-UR40M 2-2-3-1-H9-UR40M 2-2-3-1-H10-UR40M
# pair_experiment_iwslt14_2_2048_50k 2-2-3-1-H1-UR40M 2-2-3-1-H11-UR40M
# # pair_experiment_iwslt14_2_1024_50k 1-1-3-1-H12-UR40M
# pair_experiment_iwslt14_2_1024_50k 1-1-3-1-H12-UR45M
# pair_experiment_iwslt14_2_2048_50k 2-2-3-1-H1-UR40M 2-2-3-1-H11-UR40M
# pair_experiment_iwslt14_2_1024_50k 1-1-3-1-H12-UR50M
# pair_experiment_iwslt14_2_1024_50k 1-1-3-1-H12-UR20M
# pair_experiment_iwslt14_2_2048_50k 2-2-3-1-H12-UR30M
# pair_experiment_iwslt14_2_2048_50k 2-2-3-1-H12-UR20M



# bash call_scripts/train_nat.sh -e m-B-1-1-N-UR20M-rate_avg-50k \
#                                 --save-interval-updates 70000 --max-tokens 1536 \
#                                 --arch ctcpmlm_rate_selection \
#                                 --task translation_ctcpmlm \
#                                 --criterion nat_ctc_avg_rate_loss \
#                                 --has-eos --max-update 50000 \
#                                 --hydra \
#                                 -g 2 --fp16


# bash call_scripts/train_nat.sh -e m-B-3-1-N-UR20M-rate_avg-50k \
#                                 --save-interval-updates 70000 --max-tokens 1536 \
#                                 --arch ctcpmlm_rate_selection \
#                                 --task translation_ctcpmlm \
#                                 --criterion nat_ctc_avg_rate_loss \
#                                 --has-eos --max-update 50000 \
#                                 --hydra \
#                                 -g 2 --fp16

# bash call_scripts/train_nat.sh -e m-B-3-1-N-UR20M-rate_avg_1-50k \
#                                 --save-interval-updates 30000 --max-tokens 1536 \
#                                 --arch ctcpmlm_rate_selection \
#                                 --task translation_ctcpmlm \
#                                 --criterion nat_ctc_avg_rate_loss \
#                                 --has-eos --max-update 50000 \
#                                 --hydra \
#                                 --rate-list 1 \
#                                 -g 2 --fp16




experiment=t-G-3-1-N-UR30M
pair_experiment_wmt16roen_2_4096_100k_reset_meter $experiment
experiment=v-I-3-1-N-UR30M-rate_avg-33k
pair_experiment_iwslt14_2_1536_rate_avg_33k $experiment




