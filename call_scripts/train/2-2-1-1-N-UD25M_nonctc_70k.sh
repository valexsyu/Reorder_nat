source $HOME/.bashrc 
conda activate base

bash call_scripts/train_nat.sh -e 2-2-1-1-N-UD25M_nonctc_70k --save-interval-updates 10000 \
--lm-start-step 75000 --max-update 100000 \
--max-tokens 2048 -g 3 --debug \
--dryrun
# bash call_scripts/generate_nat.sh -e 2-2-1-1-N-UD25M_nonctc_70k --data-subset test-valid 