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
bash call_scripts/train_nat.sh -e  U-6-1-1-N-UF30T --fp16 -g 2 --save-interval-updates 32500 --max-update 200000 --lm-start-step 130000 --max-tokens 2048 -b 65536 --no-atten-mask --has-eos
bash call_scripts/train_nat.sh -e  Q-2-1-1-H12-UD25M --fp16 -g 2 --save-interval-updates 32500 --max-update 200000 --lm-start-step 130000 --max-tokens 2048 -b 65536 --no-atten-mask --has-eos