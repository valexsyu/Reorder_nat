#!/bin/bash

bash call_scripts/tool/watch-test-polling.sh \
    --local \
    --task translation_ctcpmlm \
    --arch nat_pretrained_model \
    --criterion nat_ctc_loss \
    -b 20 \
    --gpu_id 1 \
    -e K-2-1-1-H12-UR40M  \
    -e 2-2-1-1-H7-UF20T \
    --sleep 10

    # -e m-B-1-1-N-UR20M \

