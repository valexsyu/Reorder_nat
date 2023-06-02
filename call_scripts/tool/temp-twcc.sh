#!/bin/bash

bash call_scripts/tool/watch-test-polling.sh \
    --twcc \
    --task translation_ctcpmlm \
    --arch nat_pretrained_model \
    --criterion nat_ctc_loss \
    -b 20 \
    --gpu_id 0 \
    -e O-2-3-1-H12-UR40M \
    -e N-2-3-1-H12-UR40M \
    -e a-2-3-1-H12-UR40M \
    -e b-2-3-1-H12-UR40M \
    -e Y-2-3-1-N-UR40T \
    -e A-1-3-1-H12-UR30M \
    --sleep 10
