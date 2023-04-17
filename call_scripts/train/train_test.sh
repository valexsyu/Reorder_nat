source $HOME/.bashrc 
conda activate base

# # #16
# bash call_scripts/train_nat.sh -e m-8-1-1-A12-UF20M-test \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 1000 --lm-start-step 750 \
#                                --watch-lm-loss \
#                                --lm-iter-num 5 \
#                                -g 4 --fp16

# bash call_scripts/train_nat.sh -e p-B-3-3-B12-UF20M-lmx015-test \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 200000 --lm-start-step 150000 \
#                                 --lm-mask-rate 0.15 \
#                                 --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer \
#                                 -g 1 --fp16   

                                
# -g 2 --fp16                                   
# --lm-mask-rate 0.15

bash call_scripts/train_nat.sh -e m-B-1-1-H12-UR20M-lmx015-test \
                                --save-interval-updates 70000 --max-tokens 2048 \
                                --has-eos --max-update 100000 --lm-start-step 75000 \
                                --lm-mask-rate 0.15 \
                                --arch ctcpmlm_rate_pred \
                                --debug --dryrun 
                                # -g 1 --fp16   



  

                                                     