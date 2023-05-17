source $HOME/.bashrc 
conda activate base
# bash call_scripts/train_nat.sh -e 2-6-1-1-N-UF30T --fp16 --save-interval-updates 70000 --max-tokens 2048 --no-atten-mask
# bash call_scripts/train_nat.sh -e K-2-1-1-H12-UD25M --fp16 --save-interval-updates 70000 --max-tokens 2048 --no-atten-mask
# bash call_scripts/train_nat.sh -e F-2-1-1-H12-UD25M --fp16 -g 1 --save-interval-updates 32500 --max-update 200000 --lm-start-step 130000 --max-tokens 3276 -b 65520 --no-atten-mask
# bash call_scripts/train_nat.sh -e J-6-1-1-N-UF30T --fp16 --save-interval-updates 70000 --max-tokens 2048 --no-atten-mask
# bash call_scripts/train_nat.sh -e 2-2-1-1-H4-UR40M --fp16 --save-interval-updates 70000 --max-tokens 2048 --no-atten-mask
#bash call_scripts/train_nat.sh -e P-2-1-1-H12-UD25M --fp16 -g 1 --save-interval-updates 70000 --max-tokens 3072 --no-atten-mask
# bash call_scripts/train_nat.sh -e J-2-1-1-H12-UR40M --fp16 -g 1 --save-interval-updates 70000 --max-tokens 2048 --no-atten-mask 
# bash call_scripts/train_nat.sh -e  S-6-1-1-N-UF30T --fp16 -g 2 --save-interval-updates 32500 --max-update 200000 --lm-start-step 130000 --max-tokens 2048 -b 65536 --no-atten-mask --has-eos
# bash call_scripts/train_nat.sh -e  U-6-1-1-N-UF30T --fp16 -g 2 --save-interval-updates 32500 --max-update 200000 --lm-start-step 130000 --max-tokens 2048 -b 65536 --no-atten-mask --has-eos
# Stop excute_file=call_scripts/train/U-6-1-1-N-UF30T.sh
# excute_file=call_scripts/train/Z-6-1-1-N-UF30T.sh
# bash $excute_file || $excute_file || $excute_file || $excute_file || $excute_file || $excute_file || $excute_file || $excute_file 
# bash call_scripts/train_nat.sh -e  a-2-1-1-H12-UR40M --fp16 -g 2 --save-interval-updates 10000 --max-update 100000 \
#                                                      --lm-start-step 75000 --max-tokens 2048 -b 65536 --no-atten-mask --has-eos
# bash call_scripts/train_nat.sh -e  I-6-4-1-N-UF30T --fp16 -g 4 --save-interval-updates 70000 --max-update 100000 --lm-start-step 75000 --max-tokens 1024 -b 12288 --no-atten-mask
# bash call_scripts/train_nat.sh -e  J-6-4-1-N-UF30T --fp16 -g 4 --save-interval-updates 70000 --max-update 100000 --lm-start-step 75000 --max-tokens 1024 -b 12288 --no-atten-mask



# bash call_scripts/train_nat.sh -e m-8-3-5-K12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16


# bash call_scripts/train_nat.sh -e m-8-3-5-K12-UF20B \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 3 --fp16



# bash call_scripts/train_nat.sh -e m-8-4-5-K12-UF20B \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 3 --fp16



# bash call_scripts/train_nat.sh -e m-8-4-5-K12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 3 --fp16


# #5
# bash call_scripts/train_nat.sh -e m-8-1-1-H12-UF20M-dp001 \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16 --dropout 0.01   

# #6
# bash call_scripts/train_nat.sh -e m-8-1-1-K12-UF20M-dp001 \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16 --dropout 0.01  

                        

# #7
# bash call_scripts/train_nat.sh -e m-8-4-1-H12-UF20M-dp02 \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16 --dropout 0.2


# #8
# bash call_scripts/train_nat.sh -e m-8-4-1-K12-UF20M-dp02 \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16 --dropout 0.2 

# #7
# bash call_scripts/train_nat.sh -e m-8-4-1-H12-UF20M-dp02 \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16 --dropout 0.2




# #9
# bash call_scripts/train_nat.sh -e m-8-3-1-H12-UF20B \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16 --dropout 0.1 

# #10
# bash call_scripts/train_nat.sh -e m-8-4-1-H12-UF20B \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16 --dropout 0.1 




# #1
# bash call_scripts/train_nat.sh -e o-C-4-1-H12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16


# #2
# bash call_scripts/train_nat.sh -e o-C-4-1-K12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16



# #1
# bash call_scripts/train_nat.sh -e m-8-1-3-B12-UF20M \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 -g 2 --fp16                                


# #2
# bash call_scripts/train_nat.sh -e m-8-3-3-B12-UF20M \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 -g 2 --fp16     


# #3
# bash call_scripts/train_nat.sh -e m-B-1-3-B12-UF20M \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 -g 2 --fp16     


# #4
# bash call_scripts/train_nat.sh -e m-B-3-1-C12-UF20M \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 --watch-lm-loss \
#                                 -g 2 --fp16   



# #5
# bash call_scripts/train_nat.sh -e m-B-1-1-C12-UF20M \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 -g 2 --fp16 



# # 1
# bash call_scripts/train_nat.sh -e 2-2-1-1-H12-UF20M-eos \
#                                --save-interval-updates 70000 --max-tokens 4096 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16 



bash call_scripts/train_nat.sh -e m-B-1-1-N-UR20M-rate_select-NEW \
                                --save-interval-updates 70000 --max-tokens 3072 \
                                --arch ctcpmlm_rate_selection \
                                --task translation_ctcpmlm \
                                --criterion nat_ctc_sel_rate_loss \
                                --has-eos --max-update 100000 \
                                --hydra \
                                -g 1 --fp16

bash call_scripts/train_nat.sh -e m-B-1-1-H12-UR20M-rate_select-divTGT-NEW-2 \
                                --save-interval-updates 70000 --max-tokens 3072 \
                                --arch ctcpmlm_rate_selection \
                                --task translation_ctcpmlm \
                                --criterion nat_ctc_sel_rate_loss \
                                --has-eos --max-update 100000 \
                                --lm-start-step 25000 \
                                --hydra \
                                --debug \
                                -g 2 --fp16


function pair_experiment() { 
    bash call_scripts/train_nat.sh -e $1 \
                                    --save-interval-updates 70000 --max-tokens 3072 \
                                    --lm-start-step 75000 \
                                    --task translation_ctcpmlm \
                                    --arch nat_pretrained_model \
                                    --criterion nat_ctc_loss \
                                    --has-eos --max-update 70000 \
                                    --hydra \
                                    -g 1 --fp16       

    for experiment in $2 ; do
        mkdir checkpoints/$experiment/
        cp checkpoints/$1/checkpoint.best_bleu_* checkpoints/$experiment/
        cp checkpoints/$1/checkpoint_last.pt checkpoints/$experiment/
    done
    
    for experiment in $1 $2; do
        bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates 70000 --max-tokens 3072 \
                                        --lm-start-step 75000 \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update 100000 \
                                        --hydra \
                                        -g 1 --fp16        
    done                                                                                                                                                

}
pair_experiment 2-2-1-1-H12-UR40M 2-2-1-1-N-UR40M

bash call_scripts/train_nat.sh -e 2-2-1-1-H7-UR40M \
                                --save-interval-updates 70000 --max-tokens 3072 \
                                --lm-start-step 75000 \
                                --task translation_ctcpmlm \
                                --arch nat_pretrained_model \
                                --criterion nat_ctc_loss \
                                --has-eos --max-update 100000 \
                                --hydra \
                                -g 1 --fp16    


bash call_scripts/train_nat.sh -e m-B-1-1-N-UR20M-rate_predict_divTGT-NEW-detach \
                                --save-interval-updates 70000 --max-tokens 4096 \
                                --arch ctcpmlm_rate_predictor \
                                --task transaltion_ctcpmlm_rate \
                                --criterion nat_ctc_pred_rate_loss \
                                --hydra \
                                -g 1 --fp16   


