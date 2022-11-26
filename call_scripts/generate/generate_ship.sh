source $HOME/.bashrc 
conda activate base
# --avg-ck-turnoff 
 bash call_scripts/generate_nat.sh -b 50 --data-subset test-valid --no-atten-mask --avg-speed 1 --load-exist-bleu \
 -e 7-4-1-1-H12-UF20T \
