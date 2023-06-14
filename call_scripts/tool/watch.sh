
#1-7 
#2-10
#3-5
#4-1
#6-8

# bash call_scripts/tool/watch-test-polling.sh \
#                     --arch ctcpmlm_rate_predictor \
#                     --task transaltion_ctcpmlm_rate \
#                     --criterion nat_ctc_pred_rate_loss \
#                     -e m-B-1-1-N-UR20M-rate_predict_divTGT \
#                     -e m-B-1-1-N-UR20M-rate_predict \
#                     --sleep 100 &

# bash call_scripts/tool/watch-test-polling.sh \
#                     --arch ctcpmlm_rate_predictor \
#                     --task transaltion_ctcpmlm_rate \
#                     --criterion nat_ctc_pred_rate_loss \
#                     -e m-B-1-1-N-UR20M-rate_select \
#                     --sleep 10 &

# # execute the two commands simultaneously
# bash call_scripts/tool/watch-test-polling.sh \
#     --arch ctcpmlm_rate_predictor \
#     --task transaltion_ctcpmlm_rate \
#     --criterion nat_ctc_pred_rate_loss \
#     -e m-B-1-1-N-UR20M-rate_predict_divTGT-NEW \
#     --sleep 100 &

# bash call_scripts/tool/watch-test-polling.sh \
#     --arch ctcpmlm_rate_selection \
#     --task transaltion_ctcpmlm_rate \
#     --criterion nat_ctc_pred_rate_loss \
#     -e m-B-1-1-N-UR20M-rate_select-divTGT-NEW-2 \
#     --sleep 100 &


# # # execute the two commands simultaneously
# bash call_scripts/tool/watch-test-polling.sh \
#     --arch ctcpmlm_rate_predictor \
#     --task transaltion_ctcpmlm_rate \
#     --criterion nat_ctc_pred_rate_loss \
#     -b 20 \
#     -e m-B-1-1-N-UR20M-predsel-rate \
#     --sleep 100



#!/bin/bash

# while :
# do
#     bash call_scripts/tool/watch-test-polling-1_time.sh \
#         --arch ctcpmlm_rate_selection \
#         --task translation_ctcpmlm \
#         --criterion nat_ctc_avg_rate_loss \
#         -b 100 \
#         -e 2-2-3-1-N-UR30M-rate_avg-33k \
#         -e K-2-3-1-N-UR20M-rate_avg-33k \
#         --sleep 10

#     bash call_scripts/tool/watch-test-polling-1_time.sh \
#         --task translation_ctcpmlm \
#         --arch nat_pretrained_model \
#         --criterion nat_ctc_loss \
#         --gpu_id 0 \
#         -e t-G-3-1-N-UR30M \
#         --sleep 10
# done



bash call_scripts/tool/watch-test-polling.sh \
    --arch ctcpmlm_rate_selection \
    --task translation_ctcpmlm \
    --criterion nat_ctc_avg_rate_loss \
    -b 50 \
    --cpu \
    -e m-B-3-1-N-UR30M-rate_avg-33k_warm33 \
    --sleep 10