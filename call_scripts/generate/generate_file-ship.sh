source $HOME/.bashrc 
conda activate base

# #--------iwslt14 deen main table------
# # --avg-ck-turnoff 
# bash call_scripts/generate_nat.sh -b 1 --data-subset test-valid --no-atten-mask --avg-speed 1 \
#     -e K-2-1-1-H12-UR40M -e K-2-1-1-N-UR40M -e K-2-1-1-N-UR40T -e K-2-1-1-N-UF30T -e K-6-1-1-N-UF30T \
#     -e I-6-1-1-N-UF30T -e 2-2-1-1-H12-UR40M -e 2-2-1-1-N-UR40M -e 2-2-1-1-N-UR40T -e 2-2-1-1-N-UF30T \
#     -e 2-6-1-1-N-UF30T -e J-6-1-1-N-UF30T -e K-2-1-1-H12-UD25M -e 2-2-1-1-H12-UD25M \


#--------wmt14 deen main table------
# --avg-ck-turnoff 
bash call_scripts/generate_nat.sh -b 1 --data-subset test --no-atten-mask --avg-speed 1 \
