source $HOME/.bashrc 
conda activate base
# --avg-ck-turnoff 
 bash call_scripts/generate_nat.sh -b 1 --data-subset test --no-atten-mask --avg-speed 3 \
 -e 2-2-1-1-H12-UR25M \
 -e 2-2-1-1-H12-UD25M \
 -e 2-2-1-1-N-UF30M \