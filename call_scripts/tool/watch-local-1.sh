#!/bin/bash

bash call_scripts/tool/watch-test-polling.sh \
    --local \
    --arch ctcpmlm_rate_selection \
    --task transaltion_ctcpmlm_rate \
    --criterion nat_ctc_pred_rate_loss \
    -b 20 \
    --gpu_id 0 \
    -e 2-2-3-1-N-UR20M-rate_select-divTGT-NEW-3 \
    -e m-B-1-1-N-UR20M-rate_avg \
    --sleep 10
