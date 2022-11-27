source $HOME/.bashrc 
conda activate base
# --avg-ck-turnoff 
 bash call_scripts/generate_nat.sh -b 1 --data-subset test --no-atten-mask --avg-speed 1 \
 -e E-2-1-1-H12-UR40M \
