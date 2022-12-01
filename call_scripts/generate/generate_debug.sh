source $HOME/.bashrc 
conda activate base
# bash call_scripts/generate_nat.sh -b 50 --data-subset test-valid --avg-ck-turnoff --load-exist-bleu \
# -e J-6-1-1-N-UF30T \
# -e 2-6-1-1-N-UF30T \


# bash call_scripts/generate_nat.sh -b 50 --data-subset test-valid --avg-ck-turnoff --no-atten-mask \
# -e J-6-1-1-N-UF30T \
# -e 2-6-1-1-N-UF30T \

# bash call_scripts/generate_nat_sepcial.sh -b 50 --data-subset test-valid  --avg-ck-turnoff \
# -e E-2-1-1-H12-UF30M \

bash call_scripts/generate_nat.sh -b 1 --data-subset test-valid --no-atten-mask \
-e L-5-1-1-N-UF30T-warmup_3k-table_12 \

# --skip-exist-genfile
# --load-exist-bleu 