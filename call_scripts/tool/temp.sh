#!/bin/bash

# Create a new tmux session
tmux new-session -d -s mysession

# Split the window vertically
tmux split-window -v

# # Split the second window vertically
# tmux split-window -v
# # Split the thrid window vertically
# tmux split-window -v

# Set the size of each pane to be the same
tmux resize-pane -t 0 -y 20%
# tmux resize-pane -t 1 -y 10%
tmux resize-pane -t 2 -y 80%
# tmux resize-pane -t 3 -y 70%
# tmux resize-pane -t 2 -y 70%

Select the first window and execute the first script
tmux select-pane -t 0
tmux send-keys "bash call_scripts/tool/watch-test-best5record-twcc.sh \
    -e r-E-3-1-N-UR20M \
    -e r-E-3-1-N-UR30M \
    -e r-E-3-1-N-UR40M \
    -e m-B-3-1-N-UR30M-rate_avg-33k_warm33\
    --sleep 120" C-m

# tmux select-pane -t 1
# tmux send-keys "conda activate reorder_nat" C-m
# tmux send-keys "bash call_scripts/tool/watch-test-polling.sh \
#     --local \
#     --task translation_ctcpmlm \
#     --arch nat_pretrained_model \
#     --criterion nat_ctc_loss \
#     -b 10 \
#     --gpu_id 0 \
#     -e m-B-3-1-N-UR30M-rate_avg-33k\
#     -e m-B-3-1-N-UR30M-rate_avg \
#     -e m-B-3-1-N-UR30M-rate_avg_1 \
#     --sleep 360" C-m

# Select the second window and execute the second script
# tmux select-pane -t 1
# tmux send-keys "conda activate reorder_nat" C-m
# tmux send-keys "bash call_scripts/tool/watch-test-polling.sh \
#     --local \
#     --task translation_ctcpmlm \
#     --arch nat_pretrained_model \
#     --criterion nat_ctc_loss \
#     -b 10 \
#     --gpu_id 1 \
#     -e m-B-3-1-N-UR25M \
#     --sleep 120" C-m

# Select the third window and execute the third script
tmux select-pane -t 1
tmux send-keys "conda activate reorder_nat" C-m
tmux send-keys "bash call_scripts/tool/look_exist_best_5.sh \
    -e r-E-3-1-N-UR20M \
    -e r-E-3-1-N-UR30M \
    -e r-E-3-1-N-UR40M \
    -e m-B-3-1-N-UR30M-rate_avg-33k_warm33\
    --sleep 120" C-m

# Attach to the tmux session to view the windows
tmux attach-session -t mysession
