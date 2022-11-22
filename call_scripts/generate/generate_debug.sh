source $HOME/.bashrc 
conda activate base
# bash call_scripts/generate_nat.sh -b 10 --data-subset test-valid --avg-ck-turnoff --load-exist-bleu \
bash call_scripts/generate_nat.sh -b 50 --data-subset test-valid --no-atten-mask   \
-e E-2-1-1-H12-UF30M \
bash call_scripts/generate_nat.sh -b 200 --data-subset test-valid  --avg-ck-turnoff \
-e E-2-1-1-H12-UF30M \

# --skip-exist-genfile
# --load-exist-bleu 