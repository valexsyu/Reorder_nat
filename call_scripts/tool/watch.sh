
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

# execute the two commands simultaneously
bash call_scripts/tool/watch-test-polling.sh \
    --arch ctcpmlm_rate_predictor \
    --task transaltion_ctcpmlm_rate \
    --criterion nat_ctc_pred_rate_loss \
    -e m-B-1-1-N-UR20M-rate_predict_divTGT \
    --sleep 100 &

bash call_scripts/tool/watch-test-polling.sh \
    --task translation_ctcpmlm \
    --arch nat_pretrained_model \
    --criterion nat_ctc_loss \
    -e m-B-1-1-N-UF20M  \
    --sleep 100 &


wait
