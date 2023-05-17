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


# experiments=( "1-1-1-1-H12-UF13T" "2-2-1-1-H12-UF13T" "5-3-1-1-H12-UF13T" "7-4-1-1-H12-UF13T" )
# for experiment in "${experiments[@]}" ; do 
#         bash call_scripts/train_nat.sh \
#                 -e  $experiment --fp16 -g 1 --save-interval-updates 70000 --max-update 100000 \
#                 --lm-start-step 75000 --max-tokens 1024 -b 12288 --no-atten-mask --has-eos   
# done  
             

# #1
# bash call_scripts/train_nat.sh -e m-8-3-1-H12-UF20M-dp02 \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16 --dropout 0.2


# #2
# bash call_scripts/train_nat.sh -e m-8-3-1-K12-UF20M-dp02 \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16 --dropout 0.2

# #3
# bash call_scripts/train_nat.sh -e m-8-2-1-H12-UF20M-dp02 \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16 --dropout 0.2


# #4
# bash call_scripts/train_nat.sh -e m-8-4-1-H12-UF20M-dp02 \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16 --dropout 0.2


# #5
# bash call_scripts/train_nat.sh -e m-8-1-1-H12-UF20B \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16 --dropout 0.1


# #6
# bash call_scripts/train_nat.sh -e m-8-2-1-H12-UF20B \
#                                --save-interval-updates 70000 --max-tokens 6144 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16 --dropout 0.1

# #1
# bash call_scripts/train_nat.sh -e o-C-3-1-H12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16


# #2
# bash call_scripts/train_nat.sh -e o-C-3-1-K12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16

# #3
# bash call_scripts/train_nat.sh -e o-C-3-3-K12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16

# #4
# bash call_scripts/train_nat.sh -e o-C-3-4-K12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16  


# #5
# bash call_scripts/train_nat.sh -e o-C-1-3-K12-UR40M \
#                                --save-interval-updates 70000 --max-tokens 3072 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16


# #6
# bash call_scripts/train_nat.sh -e o-C-3-3-K12-UR40M \
#                                --save-interval-updates 70000 --max-tokens 3072 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16


#  #1
# bash call_scripts/train_nat.sh -e m-8-1-1-B12-UF20M \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 -g 2 --fp16 



#  #2
# bash call_scripts/train_nat.sh -e m-8-3-1-B12-UF20M \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 -g 2 --fp16 



#3
#bash call_scripts/train_nat.sh -e m-B-1-1-B12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                -g 2 --fp16 



# #4
# bash call_scripts/train_nat.sh -e m-B-3-1-C12-UF20M-lm5 \
#                                 --save-interval-updates 70000 --max-tokens 1536 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 --lm-iter-num 5 \
#                                 -g 4 --fp16 


# #5
# bash call_scripts/train_nat.sh -e m-B-1-1-C12-UF20M-lm5 \
#                                 --save-interval-updates 70000 --max-tokens 1536 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 --lm-iter-num 5 \
#                                 -g 4 --fp16 

# #6
# bash call_scripts/train_nat.sh -e m-B-3-1-C12-UF20M-lmx015 \
#                                 --save-interval-updates 70000 --max-tokens 3072 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 --lm-mask-rate 0.15 \
#                                 -g 2 --fp16   



# #7
# bash call_scripts/train_nat.sh -e m-B-1-1-C12-UF20M-lmx015 \
#                                 --save-interval-updates 70000 --max-tokens 3072 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 --lm-mask-rate 0.15 \
#                                 -g 2 --fp16 


# #  #8
# bash call_scripts/train_nat.sh -e m-B-1-3-A12-UF20M-lmx015 \
#                                 --save-interval-updates 70000 --max-tokens 6144 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 --lm-mask-rate 0.15 \
#                                 -g 2 --fp16 

# #  #9
# bash call_scripts/train_nat.sh -e m-B-3-3-A12-UF20M-lmx015 \
#                                 --save-interval-updates 70000 --max-tokens 6144 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 --lm-mask-rate 0.15 \
#                                 -g 2 --fp16 


# #10
# bash call_scripts/train_nat.sh -e m-B-1-3-B12-UF20B-lmx015 \
#                                 --save-interval-updates 70000 --max-tokens 6144 \
#                                 --has-eos --max-update 100000 --lm-start-step 75000 \
#                                 --lm-mask-rate 0.15 \
#                                 -g 2 --fp16  


# #11
# bash call_scripts/train_nat.sh -e m-B-1-3-C12-UF20M-lm5 \
#                             --save-interval-updates 70000 --max-tokens 6144 \
#                             --has-eos --max-update 100000 --lm-start-step 75000 \
#                             --lm-iter-num 5 \
#                             -g 2 --fp16  


# #2-12
# bash call_scripts/train_nat.sh -e m-B-1-3-B12-UF20B \
#                             --save-interval-updates 70000 --max-tokens 6144 \
#                             --has-eos --max-update 100000 --lm-start-step 75000 \
#                             --lm-iter-num 1 \
#                             -g 2 --fp16 


# #2-13
# bash call_scripts/train_nat.sh -e m-B-3-3-B12-UF20M \
#                             --save-interval-updates 70000 --max-tokens 6144 \
#                             --has-eos --max-update 100000 --lm-start-step 75000 \
#                             --lm-iter-num 1 \
#                             -g 2 --fp16 



# bash call_scripts/train_nat.sh -e m-B-1-1-H12-UF20M   \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 \
#                                 -g 3 --fp16  

# bash call_scripts/train_nat.sh -e m-B-3-1-N-UF20M   \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 \
#                                 -g 3 --fp16  



# bash call_scripts/train_nat.sh -e m-B-1-1-N-UF20M \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --task translation_ctcpmlm \
#                                 --arch nat_pretrained_model \
#                                 --criterion nat_ctc_loss \
#                                 --has-eos --max-update 100000 \
#                                 --hydra \
#                                 -g 2 --fp16   

# bash call_scripts/train_nat.sh -e m-B-1-1-N-UF20M-nohydra \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --task translation_ctcpmlm \
#                                 --arch nat_pretrained_model \
#                                 --criterion nat_ctc_loss \
#                                 --has-eos --max-update 100000 \
#                                 -g 2 --fp16   


# bash call_scripts/train_nat.sh -e m-B-1-1-N-UF20M-NEW \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --task translation_ctcpmlm \
#                                 --arch nat_pretrained_model \
#                                 --criterion nat_ctc_loss \
#                                 --has-eos --max-update 100000 \
#                                 --hydra \
#                                 -g 2 --fp16       

# bash call_scripts/train_nat.sh -e m-B-1-1-N-UR30M-NEW \
#                                 --save-interval-updates 70000 --max-tokens 1536 \
#                                 --task translation_ctcpmlm \
#                                 --arch nat_pretrained_model \
#                                 --criterion nat_ctc_loss \
#                                 --has-eos --max-update 100000 \
#                                 --hydra \
#                                 -g 2 --fp16       

# bash call_scripts/train_nat.sh -e m-B-1-1-N-UR40M-NEW \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --task translation_ctcpmlm \
#                                 --arch nat_pretrained_model \
#                                 --criterion nat_ctc_loss \
#                                 --has-eos --max-update 100000 \
#                                 --hydra \
#                                 -g 2 --fp16    





function pair_experiment() { 
    bash call_scripts/train_nat.sh -e $1 \
                                    --save-interval-updates 70000 --max-tokens 3072 \
                                    --lm-start-step 75000 \
                                    --task translation_ctcpmlm \
                                    --arch nat_pretrained_model \
                                    --criterion nat_ctc_loss \
                                    --has-eos --max-update 70000 \
                                    --hydra \
                                    -g 2 --fp16       


        for experiment in $2 ; do
            if [ -e checkpoints/$experiment/checkpoint_last.pt ] && \
            [ $(ls checkpoints/$experiment/checkpoint.best_bleu_* 2>/dev/null | grep -c "^checkpoints/$experiment/checkpoint.best_bleu_.*") -eq 5 ]; then
                echo "All 6 checkpoint files exist"
            else        
                mkdir checkpoints/$experiment/
                cp checkpoints/$1/checkpoint.best_bleu_* checkpoints/$experiment/
                cp checkpoints/$1/checkpoint_last.pt checkpoints/$experiment/
            fi
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
                                        -g 2 --fp16        
    done                                                                                                                                                

}

pair_experiment m-B-3-1-H12-UF40M m-B-3-1-N-UF40M 



function pair_experiment_1() { 
    bash call_scripts/train_nat.sh -e $1 \
                                    --save-interval-updates 70000 --max-tokens 2048 \
                                    --lm-start-step 75000 \
                                    --task translation_ctcpmlm \
                                    --arch nat_pretrained_model \
                                    --criterion nat_ctc_loss \
                                    --has-eos --max-update 70000 \
                                    --hydra \
                                    -g 2 --fp16       


        for experiment in $2 ; do
            if [ -e checkpoints/$experiment/checkpoint_last.pt ] && \
            [ $(ls checkpoints/$experiment/checkpoint.best_bleu_* 2>/dev/null | grep -c "^checkpoints/$experiment/checkpoint.best_bleu_.*") -eq 5 ]; then
                echo "All 6 checkpoint files exist"
            else        
                mkdir checkpoints/$experiment/
                cp checkpoints/$1/checkpoint.best_bleu_* checkpoints/$experiment/
                cp checkpoints/$1/checkpoint_last.pt checkpoints/$experiment/
            fi
        done

    for experiment in $1 $2; do
        bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates 70000 --max-tokens 2048 \
                                        --lm-start-step 75000 \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update 100000 \
                                        --hydra \
                                        -g 2 --fp16        
    done                                                                                                                                                

}

pair_experiment_1 K-2-1-1-H7-UR40M m-B-3-1-N-UF40M K-2-1-1-H12-UR40M

