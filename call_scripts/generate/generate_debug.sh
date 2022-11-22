source $HOME/.bashrc 
conda activate base
bash call_scripts/generate_nat.sh -b 200 --data-subset test-valid --avg-ck-turnoff --no-atten-mask  \
-e 2-2-1-1-N-UD13T \

# --skip-exist-genfile
# --load-exist-bleu 