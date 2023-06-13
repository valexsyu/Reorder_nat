#!/bin/bash

while :
do
    bash call_scripts/tool/watch-test-polling-1_time.sh \
        --twcc \
        --arch ctcpmlm_rate_selection \
        --task translation_ctcpmlm \
        --criterion nat_ctc_avg_rate_loss \
        --gpu_id 0 \
        -b 50 \
        -e r-E-3-1-N-UR30M-rate_avg-33k \
        -e s-F-3-1-N-UR30M-rate_avg-33k \
        --sleep 10

    bash call_scripts/tool/watch-test-polling-1_time.sh \
        --twcc \
        --task translation_ctcpmlm \
        --arch nat_pretrained_model \
        --criterion nat_ctc_loss \
        -b 50 \
        --gpu_id 0 \
        -e r-E-3-1-N-UR30M \
        -e r-E-3-1-N-UR40M \
        --sleep 10
done

# while :
# do

#     bash call_scripts/tool/watch-test-polling-1_time.sh \
#         --twcc \
#         --task translation_ctcpmlm \
#         --arch nat_pretrained_model \
#         --criterion nat_ctc_loss \
#         -b 50 \
#         --gpu_id 0 \
#         -e r-E-3-1-N-UR20M \
#         -e r-E-3-1-N-UR30M \
#         -e r-E-3-1-N-UR40M \
#         --sleep 10

#     bash call_scripts/tool/watch-test-polling-1_time.sh \
#         --twcc \
#         --arch ctcpmlm_rate_selection \
#         --task translation_ctcpmlm \
#         --criterion nat_ctc_avg_rate_loss \
#         -b 50 \
#         --gpu_id 0 \
#         -e m-B-3-1-N-UR30M-rate_avg-33k_warm33 \
#         --sleep 10

# done