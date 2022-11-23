# CUDA_VISIBLE_DEVICES=0 bash call_scripts/generate_nat.sh -b 50 --data-subset test-valid --no-atten-mask --twcc \
bash call_scripts/generate_nat.sh -b 50 --data-subset test-valid --no-atten-mask --twcc --avg-ck-turnoff \
-e 1-1-1-1-N-UF20T \
-e 1-1-1-1-H12-UF20T \
 
# CUDA_VISIBLE_DEVICES=0 bash call_scripts/generate_nat.sh -b 100 --data-subset test-valid --avg-ck-turnoff --twcc \
# -e 1-1-1-1-N-UF20T \
# -e 1-1-1-1-H12-UF20T \
