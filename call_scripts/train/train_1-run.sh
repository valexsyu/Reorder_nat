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


bash call_scripts/train_nat.sh -e m-8-1-1-K12-UF20M-AutoModelForMaskedLM-randPos  \
                               --save-interval-updates 70000 --max-tokens 4096 \
                               --has-eos --max-update 100000 --lm-start-step 75000 \
                               --g 1 --fp16


bash call_scripts/train_nat.sh -e m-8-3-1-K12-UR40M-AutoModelForMaskedLM-randPos  \
                               --save-interval-updates 70000 --max-tokens 3072 \
                               --has-eos --max-update 100000 --lm-start-step 75000 \
                               --g 1 --fp16









