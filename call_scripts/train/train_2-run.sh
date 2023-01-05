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
bash call_scripts/train_nat.sh -e  Y-6-1-1-N-UF30T --fp16 -g 2 --save-interval-updates 10000 --max-update 100000 --lm-start-step 75000 --max-tokens 2048 -b 65536 --no-atten-mask --has-eos
bash call_scripts/train_nat.sh -e  Z-6-1-1-N-UF30T --fp16 -g 2 --save-interval-updates 10000 --max-update 100000 --lm-start-step 75000 --max-tokens 2048 -b 65536 --no-atten-mask --has-eos
bash call_scripts/train_nat.sh -e  a-2-1-1-H12-UR40M --fp16 -g 2 --save-interval-updates 10000 --max-update 100000 \
                                                     --lm-start-step 75000 --max-tokens 2048 -b 65536 --no-atten-mask --has-eos

                                                     