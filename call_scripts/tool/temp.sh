#!/bin/bash

# Create a new tmux session
tmux new-session -d -s mysession

# Split the window vertically
tmux split-window -v

# #=============2==================
# #Set the size of each pane to be the same
# tmux resize-pane -t 0 -y 20%
# tmux resize-pane -t 1 -y 80%

# #==========3==================
# Split the second window vertically
tmux split-window -v
tmux resize-pane -t 0 -y 10%
tmux resize-pane -t 1 -y 10%
tmux resize-pane -t 2 -y 70%

# #============4==================
# # Split the second window vertically
# tmux split-window -v
# # Split the thrid window vertically
# tmux split-window -v
# tmux resize-pane -t 0 -y 10%
# tmux resize-pane -t 1 -y 10%
# tmux resize-pane -t 2 -y 10%
# tmux resize-pane -t 3 -y 70%




Select the first window and execute the first script
tmux select-pane -t 0
tmux send-keys "bash call_scripts/tool/watch-test-best5record-twcc.sh \
    -e s-F-3-1-N-UR30M-rate_avg-33k_100k \
    -e s-F-3-1-N-UR20M-100k_300k \
    -e s-F-3-1-N-UR30M-100k_300k \
    -e s-F-3-1-N-UR40M-100k_300k \
    --sleep 120" C-m

tmux select-pane -t 1
tmux send-keys "conda activate reorder_nat" C-m
# tmux send-keys "bash call_scripts/tool/watch-test-polling.sh \
#     --local \
#     --task translation_ctcpmlm \
#     --arch nat_pretrained_model \
#     --criterion nat_ctc_loss \
#     -b 10 \
#     --gpu_id 1 \
#     -e K-2-3-1-N-UR30M-rate_avg-100k \
#     -e 2-2-3-1-N-UR30M-rate_avg-100k \
#     --sleep 360" C-m

Select the second window and execute the second script
tmux select-pane -t 1
tmux send-keys "conda activate reorder_nat" C-m
tmux send-keys "bash call_scripts/tool/watch-test-polling.sh \
    --local \
    --task translation_ctcpmlm \
    --arch nat_pretrained_model \
    --criterion nat_ctc_loss \
    -b 10 \
    --gpu_id 1 \
    -e 2-2-3-1-N-UR20M \
    -e 2-2-3-1-N-UR30M \
    -e 2-2-3-1-N-UR40M \
    -e K-2-3-1-N-UR20M \
    -e K-2-3-1-N-UR30M
    --sleep 120" C-m

# Select the third window and execute the third script
tmux select-pane -t 2
tmux send-keys "conda activate reorder_nat" C-m
tmux send-keys "bash call_scripts/tool/look_exist_best_5.sh \
    -e s-F-3-1-N-UR30M-rate_avg-33k_100k \
    -e s-F-3-1-N-UR20M-100k_300k \
    -e s-F-3-1-N-UR30M-100k_300k \
    -e s-F-3-1-N-UR40M-100k_300k \
    -e 2-2-3-1-N-UR20M \
    -e 2-2-3-1-N-UR30M \
    -e 2-2-3-1-N-UR40M \
    -e K-2-3-1-N-UR20M \
    -e K-2-3-1-N-UR30M \
    --sleep 120" C-m

# Attach to the tmux session to view the windows
tmux attach-session -t mysession
