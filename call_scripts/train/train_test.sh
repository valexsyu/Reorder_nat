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

# bash call_scripts/train_nat.sh -e m-B-1-1-N-UR20M-GGGGGGGGGGGGGGGGGGGGGGGGGGG \
#                                 --save-interval-updates 70000 --max-tokens 512 \
#                                 --arch ctcpmlm_rate_predictor \
#                                 --task transaltion_ctcpmlm_rate \
#                                 --criterion nat_ctc_pred_rate_loss \
#                                 --hydra \
#                                 --local \
#                                 --valid-set \
#                                 -g 1 --fp16






# fairseq-hydra-train -r --config-dir checkpoints/m-B-1-1-N-UR20M-test/  --config-name m-B-1-1-N-UR20M.yaml 
  

# bash call_scripts/train_nat.sh -e m-B-1-1-N-UR20M-TTTTTTT \
#                                 --save-interval-updates 70000 --max-tokens 768 \
#                                 --arch ctcpmlm_rate_selection \
#                                 --task translation_ctcpmlm \
#                                 --criterion nat_ctc_sel_rate_loss \
#                                 --has-eos --max-update 100000 \
#                                 --hydra \
#                                 --debug \
#                                 --valid-set --local \
#                                 -g 1 --fp16

# CUDA_VISIBLE_DEVICES=0 bash call_scripts/train_nat.sh -e 2-2-1-1-H12-UF20M-TTTTTTTTTT \
#                                     --save-interval-updates 70000 --max-tokens 1024 \
#                                     --lm-start-step 75000 \
#                                     --task translation_ctcpmlm \
#                                     --arch nat_pretrained_model \
#                                     --criterion nat_ctc_loss \
#                                     --has-eos --max-update 100000 \
#                                     --local \
#                                     --dryrun \
#                                     --hydra \
#                                     -g 1 --fp16                                   
# function pair_experiment() { 
#     CUDA_VISIBLE_DEVICES=0 bash call_scripts/train_nat.sh -e $1 \
#                                     --save-interval-updates 70000 --max-tokens 1024 \
#                                     --lm-start-step 75000 \
#                                     --task translation_ctcpmlm \
#                                     --arch nat_pretrained_model \
#                                     --criterion nat_ctc_loss \
#                                     --has-eos --max-update 100 \
#                                     --local \
#                                     --hydra \
#                                     --valid-set \
#                                     -g 1 --fp16       

#     for experiment in $2; do
#         mkdir checkpoints/$experiment/
#         cp checkpoints/$1/checkpoint.best_bleu_* checkpoints/$experiment/
#         cp checkpoints/$1/checkpoint_last.pt checkpoints/$experiment/
#     done
    
#     for experiment in $1 $2; do
#         CUDA_VISIBLE_DEVICES=0 bash call_scripts/train_nat.sh -e $experiment \
#                                 --save-interval-updates 70000 --max-tokens 1024 \
#                                 --lm-start-step 75000 \
                                # --task translation_ctcpmlm \
                                # --arch nat_pretrained_model \
                                # --criterion nat_ctc_loss \
#                                 --has-eos --max-update 120 \
#                                 --hydra \
#                                 --local \
#                                 --valid-set \
#                                 -g 1 --fp16        
#     done                                                                                                                                                

# }

# pair_experiment 2-2-1-1-H12-UF20M-TTTTTTT 2-2-1-1-N-UF20M-TTTTTTTTTT 




# bash call_scripts/train_nat.sh -e 2-2-1-1-H12-UR40M-TTTTTTTTTTTTTTGGGG  \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --task translation_ctcpmlm \
#                                 --arch nat_pretrained_model \
#                                 --criterion nat_ctc_loss \
#                                 --no-atten-mask \
#                                 --debug \
#                                 --hydra \
#                                 --valid-set
#                                 # -g 1 --fp16  


bash call_scripts/train_nat.sh -e m-B-1-1-N-UR20M-rate_TTTTTTTTTTTTTTTTTTT \
                                --save-interval-updates 70000 --max-tokens 1024 \
                                --arch ctcpmlm_rate_selection \
                                --task translation_ctcpmlm \
                                --criterion nat_ctc_avg_rate_loss \
                                --has-eos --max-update 50000 \
                                --hydra \
                                --rate-list 1 \
                                --dryrun \
                                -g 4 --fp16  