source $HOME/.bashrc 
conda activate base
# bash call_scripts/generate_nat.sh -b 10 --data-subset test-valid --avg-ck-turnoff --load-exist-bleu \
bash call_scripts/generate_nat.sh -b 50 --data-subset test-valid --no-atten-mask --avg-ck-turnoff \
-e 2-2-1-1-H12-UR15M \

# --skip-exist-genfile
# --load-exist-bleu 