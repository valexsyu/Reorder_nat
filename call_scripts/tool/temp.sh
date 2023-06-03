#!/bin/bash

# Create a new tmux session
tmux new-session -d -s mysession

# Split the window vertically
tmux split-window -v

# Split the second window vertically
tmux split-window -v
# Split the thrid window vertically
# tmux split-window -v

# Set the size of each pane to be the same
# tmux resize-pane -t 0 -y 15%
tmux resize-pane -t 0 -y 15%
tmux resize-pane -t 1 -y 15%
tmux resize-pane -t 2 -y 70%

# Select the first window and execute the first script
# tmux select-pane -t 0
# tmux send-keys "bash call_scripts/tool/watch-test-best5record-twcc.sh \
#     -e O-2-3-1-H12-UR40M \
#     -e N-2-3-1-H12-UR40M \
#     -e a-2-3-1-H12-UR40M \
#     -e b-2-3-1-H12-UR40M \
#     -e Y-2-3-1-N-UR40T \
#     -e A-1-3-1-H12-UR30M \
#     -e B-1-3-1-H12-UR30M \
#     -e M-1-3-1-H12-UR30M \
#     -e L-1-3-1-H12-UR30M \
#     -e M-5-3-1-N-UF30T \
#     -e V-5-3-1-N-UF30T \
#     -e V-1-3-1-N-UF30T \
#     -e V-1-3-1-N-UR30T \
#     -e L-5-3-1-N-UF30T \
#     -e C-5-3-1-N-UF30T \
#     -e C-1-3-1-N-UF30T \
#     -e C-1-3-1-N-UR30T \
#     --sleep 10" C-m

tmux select-pane -t 0
tmux send-keys "conda activate reorder_nat" C-m
tmux send-keys "bash call_scripts/tool/watch-test-polling.sh \
    --local \
    --task translation_ctcpmlm \
    --arch nat_pretrained_model \
    --criterion nat_ctc_loss \
    -b 20 \
    --gpu_id 0 \
    -e 1-1-3-1-H12-UR15MT \
    -e 1-1-3-1-H12-UR20M \
    -e 1-1-3-1-H12-UR22M \
    -e 1-1-3-1-H12-UR30M \
    -e 1-1-3-1-H12-UR33M \
    -e 1-1-3-1-H12-UR40M \
    -e 1-1-3-1-H12-UR45M \
    -e 1-1-3-1-H12-UR50M \
    --sleep 120" C-m

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
    -e Z-6-3-1-N-UF30T \
    -e 2-2-3-1-H5-UR40M \
    -e 2-2-3-1-H6-UR40M \
    -e 2-2-3-1-H7-UR40M \
    -e 2-2-3-1-H8-UR40M \
    -e 2-2-3-1-H9-UR40M \
    -e 2-2-3-1-H10-UR40M \
    -e 2-2-3-1-H11-UR40M \
    -e O-2-3-1-H12-UR40M \
    -e N-2-3-1-H12-UR40M \
    -e a-2-3-1-H12-UR40M \
    -e b-2-3-1-H12-UR40M \
    -e Y-2-3-1-N-UR40T \
    --sleep 120" C-m

# Select the third window and execute the third script
tmux select-pane -t 2
tmux send-keys "conda activate reorder_nat" C-m
tmux send-keys "bash call_scripts/tool/look_exist_best_5.sh \
    -e Z-6-3-1-N-UF30T \
    -e 2-2-3-1-H5-UR40M \
    -e 2-2-3-1-H6-UR40M \
    -e 2-2-3-1-H7-UR40M \
    -e 2-2-3-1-H8-UR40M \
    -e 2-2-3-1-H9-UR40M \
    -e 2-2-3-1-H10-UR40M \
    -e 2-2-3-1-H11-UR40M \
    -e O-2-3-1-H12-UR40M \
    -e N-2-3-1-H12-UR40M \
    -e a-2-3-1-H12-UR40M \
    -e b-2-3-1-H12-UR40M \
    -e Y-2-3-1-N-UR40T \
    -e 1-1-3-1-H12-UR15MT \
    -e 1-1-3-1-H12-UR20M \
    -e 1-1-3-1-H12-UR22M \
    -e 1-1-3-1-H12-UR30M \
    --sleep 120" C-m

# Attach to the tmux session to view the windows
tmux attach-session -t mysession
