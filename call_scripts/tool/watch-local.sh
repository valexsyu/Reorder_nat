source $HOME/.bashrc 
conda activate base
bash call_scripts/tool/watch-test-polling.sh \
    --arch ctcpmlm_rate_selection \
    --task transaltion_ctcpmlm_rate \
    --criterion nat_ctc_pred_rate_loss \
    --local \
    -e m-B-1-1-N-UR20M-rate_select-NEW \
    -e m-B-1-1-N-UR20M-rate_select-divTGT-NEW \
    --sleep 10