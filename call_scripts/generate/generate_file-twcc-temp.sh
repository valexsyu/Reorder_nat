
# CUDA_VISIBLE_DEVICES=0 bash call_scripts/generate_nat.sh -b 50 --data-subset test-valid --no-atten-mask --twcc \
# bash call_scripts/generate_nat.sh -b 50 --data-subset test-valid --no-atten-mask --twcc --avg-ck-turnoff \
# -e 2-2-1-1-H12-UR40M \
# -e 2-2-1-1-H12-UR20M \

CUDA_VISIBLE_DEVICES=0 bash call_scripts/generate_nat.sh -b 100 --data-subset test-valid --avg-ck-turnoff --twcc \
-e 2-2-1-1-H12-UR40M \
-e 2-2-1-1-H12-UR20M \
