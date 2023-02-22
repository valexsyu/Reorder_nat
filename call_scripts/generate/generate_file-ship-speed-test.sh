source $HOME/.bashrc 
conda activate base

# hrun -N s01 -G -c 8 -m 20 bash call_scripts/generate_nat.sh -b 1 \
#                 --data-subset test --ck-types top --avg-speed 1 --no-atten-mask \
#                 --skip-load-step-num \
# -e 2-2-1-1-H12-UD30M 


bash call_scripts/generate_nat.sh -b 1 \
    --data-subset test --ck-types top --avg-speed 5 --no-atten-mask --avg-ck-turnoff \
    -e J-6-4-1-N-UF30T \
    -e 2-6-4-1-N-UF30T \
    -e 2-2-1-1-N-UF30T \
    -e 2-2-1-1-N-UR40T \
    -e 2-2-1-1-N-UR40M \
    -e 2-2-1-1-H12-UR40M \
    -e b-6-4-1-N-UF30T \
    -e Z-6-4-1-N-UF30T \
    -e Z-2-1-1-N-UF30T \
    -e Z-2-1-1-N-UR40T \
    -e Z-2-1-1-N-UR40M \
    -e Z-2-1-1-H12-UR40M \
    -e 1-5-4-1-H12-UF20T \
    -e 1-1-1-1-H12-UF20T \
    -e 2-2-1-1-H12-UF20T \
    -e 5-3-1-1-H12-UF20T \
    -e 7-4-1-1-H12-UF20T \
    -e 2-2-1-1-H12-UR50M \
    -e 2-2-1-1-H12-UR45M \
    -e 2-2-1-1-H12-UR40M \
    -e 2-2-1-1-H12-UR33M \
    -e 2-2-1-1-H12-UR30M \
    -e 2-2-1-1-H12-UR25M \
    -e 2-2-1-1-H12-UR22M \
    -e 2-2-1-1-H12-UR20M \
    -e 2-2-1-1-H12-UR15M \
    -e 2-2-1-1-H12-UD50M \
    -e 2-2-1-1-H12-UD45M \
    -e 2-2-1-1-H12-UD40M \
    -e 2-2-1-1-H12-UD33M \
    -e 2-2-1-1-H12-UD30M \
    -e 2-2-1-1-H12-UD25M \
    -e 2-2-1-1-H12-UD22M \
    -e 2-2-1-1-H12-UD20M \
    -e 2-2-1-1-H12-UD15M 


LM_PATH=/livingrooms/valexsyu/dataset/nat/4-gram-lm/2-4-gram-lm.bin
bash call_scripts/generate_nat_jcx.sh -e 2-2-1-1-H12-UR40M -b 1 \
    --avg-ck-turnoff --ck-types top --data-subset test \
    --ctc-beam-decoding \
    --beam-size 20 \
    --kenlm-path $LM_PATH \
    --alpha 0.3 \
    --beta 0.9 \
    --avg-speed 5

LM_PATH=/livingrooms/valexsyu/dataset/nat/4-gram-lm/Z-4-gram-lm.bin
bash call_scripts/generate_nat_jcx.sh -e Z-2-1-1-H12-UR40M -b 1 \
    --avg-ck-turnoff --ck-types top --data-subset test \
    --ctc-beam-decoding \
    --beam-size 20 \
    --kenlm-path $LM_PATH \
    --alpha 0.3 \
    --beta 0.9 \
    --avg-speed 5
	