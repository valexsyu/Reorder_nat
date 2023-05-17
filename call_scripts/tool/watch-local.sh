#!/bin/bash

bash call_scripts/tool/watch-test-polling.sh \
    --local \
    --task translation_ctcpmlm \
    --arch nat_pretrained_model \
    --criterion nat_ctc_loss \
    -b 20 \
    --gpu_id 0 \
    -e m-B-3-1-N-UF40M  \
    -e m-B-1-1-N-UR20M \
    --sleep 10


