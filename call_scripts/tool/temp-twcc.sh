#!/bin/bash

bash call_scripts/tool/watch-test-polling.sh \
    --twcc \
    --task translation_ctcpmlm \
    --arch nat_pretrained_model \
    --criterion nat_ctc_loss \
    -b 50 \
    --gpu_id 0 \
    -e O-2-3-1-H12-UR40M \
    -e N-2-3-1-H12-UR40M \
    -e a-2-3-1-H12-UR40M \
    -e b-2-3-1-H12-UR40M \
    -e Y-2-3-1-N-UR40T \
    -e A-1-3-1-H12-UR30M \
    -e B-1-3-1-H12-UR30M \
    -e M-1-3-1-H12-UR30M \
    -e L-1-3-1-H12-UR30M \
    -e M-5-3-1-N-UF30T \
    -e V-5-3-1-N-UF30T \
    -e V-1-3-1-N-UF30T \
    -e V-1-3-1-N-UR30T \
    -e L-5-3-1-N-UF30T \
    -e C-5-3-1-N-UF30T \
    -e C-1-3-1-N-UF30T \
    -e C-1-3-1-N-UR30T \
    --sleep 10
