CUDA_VISIBLE_DEVICES=0 bash call_scripts/generate_nat.sh -b 50 --data-subset test-valid --no-atten-mask --twcc \
-e 2-2-2-1-H12-UF20T \
-e 2-2-2-1-N-UF20T \

CUDA_VISIBLE_DEVICES=0 bash call_scripts/generate_nat.sh -b 60 --data-subset test-valid --avg-ck-turnoff --twcc \
-e 2-2-2-1-H12-UF20T \
-e 2-2-2-1-N-UF20T \
