source $HOME/.bashrc 
conda activate base
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
bash call_scripts/train_nat.sh -e  b-2-1-1-H12-UR40M --fp16 -g 2 --save-interval-updates 10000 --max-update 100000 \
                                                    --lm-start-step 75000 --max-tokens 2048 -b 65536 --no-atten-mask --has-eos
bash call_scripts/train_nat.sh -e  K-6-4-1-N-UF30T --fp16 -g 2 --save-interval-updates 70000 --max-update 100000 --lm-start-step 75000 --max-tokens 3072 -b 12288 --no-atten-mask 
bash call_scripts/train_nat.sh -e  I-6-4-1-N-UF30T --fp16 -g 2 --save-interval-updates 70000 --max-update 100000 --lm-start-step 75000 --max-tokens 3072 -b 12288 --no-atten-mask



