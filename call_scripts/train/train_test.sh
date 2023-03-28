source $HOME/.bashrc 
conda activate base

#16
bash call_scripts/train_nat.sh -e m-8-1-1-A12-UF20M-test \
                               --save-interval-updates 70000 --max-tokens 2048 \
                               --has-eos --max-update 1000 --lm-start-step 750 \
                               --watch-lm-loss \
                               --lm-iter-num 5 \
                               -g 4 --fp16








  

                                                     