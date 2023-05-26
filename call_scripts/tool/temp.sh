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
# tmux select-pane -t 0
# tmux send-keys "conda activate reorder_nat" C-m
# tmux send-keys "bash call_scripts/tool/watch-test-polling.sh \
#     --local \
#     --arch ctcpmlm_rate_predictor \
#     --task transaltion_ctcpmlm_rate \
#     --criterion nat_ctc_pred_rate_loss \
#     -b 20 \
#     --gpu_id 1 \
#     -e m-B-1-1-N-UR20M-predsel-rate \
#     --sleep 100" C-m


tmux select-pane -t 0
tmux send-keys "conda activate reorder_nat" C-m
tmux send-keys "bash call_scripts/tool/watch-test-polling.sh \
    --local \
    --task translation_ctcpmlm \
    --arch nat_pretrained_model \
    --criterion nat_ctc_loss \
    -b 20 \
    --gpu_id 1 \
    -e 2-6-3-1-N-UF30T \
    -e 2-2-3-1-N-UF30T \
    -e K-2-3-1-N-UR40M \
    -e K-2-3-1-H12-UR40M  \
    -e 2-2-3-1-N-UR40T \
    -e 2-2-3-1-H12-UR40T \
    --sleep 20" C-m


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
    -e K-6-3-1-N-UF30T \
    -e K-2-3-1-N-UF30T \
    -e J-6-3-1-N-UF30T \
    -e I-6-3-1-N-UF30T \
    --sleep 5" C-m

# Select the third window and execute the third script
tmux select-pane -t 2
tmux send-keys "conda activate reorder_nat" C-m
tmux send-keys "bash call_scripts/tool/look_exist_best_5.sh \
    -e 2-6-3-1-N-UF30T \
    -e 2-2-3-1-N-UF30T \
    -e K-2-3-1-N-UR40M \
    -e K-2-3-1-H12-UR40M \
    -e m-B-1-1-N-UR20M-predsel-rate \
    -e 2-2-3-1-N-UR40T \
    -e 2-2-3-1-H12-UR40T \    
    -e K-6-3-1-N-UF30T \
    -e K-2-3-1-N-UF30T \
    -e J-6-3-1-N-UF30T \
    -e I-6-3-1-N-UF30T \
    --sleep 120" C-m

# Attach to the tmux session to view the windows
tmux attach-session -t mysession
