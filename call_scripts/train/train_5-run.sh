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
