#!/bin/bash

# Create a new tmux session
tmux new-session -d -s mysession

# Split the window vertically
tmux split-window -v

# Split the second window vertically
tmux split-window -v

# Set the size of each pane to be the same
tmux resize-pane -t 0 -y 20%
tmux resize-pane -t 1 -y 20%
tmux resize-pane -t 2 -y 60%

# Select the first window and execute the first script
tmux select-pane -t 0
tmux send-keys "conda activate reorder_nat" C-m
tmux send-keys "bash call_scripts/tool/watch-test-polling.sh \
    --local \
    --arch ctcpmlm_rate_predictor \
    --task transaltion_ctcpmlm_rate \
    --criterion nat_ctc_pred_rate_loss \
    -b 20 \
    --gpu_id 0 \
    -e m-B-1-1-N-UR20M-predsel-rate \
    --sleep 100" C-m

# Select the second window and execute the second script
tmux select-pane -t 1
tmux send-keys "conda activate reorder_nat" C-m
tmux send-keys "bash call_scripts/tool/watch-test-polling.sh \
    --local \
    --task translation_ctcpmlm \
    --arch nat_pretrained_model \
    --criterion nat_ctc_loss \
    -b 20 \
    --gpu_id 1 \
    -e 2-6-1-1-H7-UF20M  \
    -e 2-2-1-1-H7-UF20T \
    -e 2-2-1-1-H7-UF20T \
    -e J-6-1-1-H7-UF20M \
    --sleep 60" C-m

# Select the third window and execute the third script
tmux select-pane -t 2
tmux send-keys "conda activate reorder_nat" C-m
tmux send-keys "bash call_scripts/tool/look_exist_best_5.sh \
    -e m-B-1-1-N-UR20M-predsel-rate \
    -e 2-6-1-1-H7-UF20M \
    -e 2-2-1-1-H7-UF20T \
    -e 2-2-1-1-H7-UF20T \
    -e J-6-1-1-H7-UF20M \
    --sleep 60" C-m

# Attach to the tmux session to view the windows
tmux attach-session -t mysession
