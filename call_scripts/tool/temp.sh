#!/bin/bash

# Create a new tmux session
tmux new-session -d -s mysession

# Split the window vertically
tmux split-window -v

# Split the second window vertically
tmux split-window -v
# Split the thrid window vertically
tmux split-window -v

# Set the size of each pane to be the same
tmux resize-pane -t 0 -y 15%
tmux resize-pane -t 1 -y 15%
tmux resize-pane -t 2 -y 15%
tmux resize-pane -t 3 -y 55%

Select the first window and execute the first script
tmux select-pane -t 0
tmux send-keys "bash call_scripts/tool/watch-test-polling-twcc.sh \
    -e O-2-3-1-H12-UR40M \
    -e N-2-3-1-H12-UR40M \
    -e a-2-3-1-H12-UR40M \
    -e b-2-3-1-H12-UR40M \
    -e Y-2-3-1-N-UR40T \
    --sleep 100" C-m

tmux select-pane -t 1
tmux send-keys "conda activate reorder_nat" C-m
tmux send-keys "bash call_scripts/tool/watch-test-polling.sh \
    --local \
    --task translation_ctcpmlm \
    --arch nat_pretrained_model \
    --criterion nat_ctc_loss \
    -b 20 \
    --gpu_id 0 \
    -e Z-6-3-1-N-UF30T \
    -e O-2-3-1-H12-UR40M \
    -e N-2-3-1-H12-UR40M \
    -e a-2-3-1-H12-UR40M \
    -e b-2-3-1-H12-UR40M \
    -e Y-2-3-1-N-UR40T \
    --sleep 120" C-m


# Select the second window and execute the second script
tmux select-pane -t 2
tmux send-keys "conda activate reorder_nat" C-m
tmux send-keys "bash call_scripts/tool/watch-test-polling.sh \
    --local \
    --task translation_ctcpmlm \
    --arch nat_pretrained_model \
    --criterion nat_ctc_loss \
    -b 20 \
    --gpu_id 1 \
    -e J-2-3-1-H12-UR40T \
    -e b-6-3-1-N-UF30T \
    --sleep 120" C-m

# Select the third window and execute the third script
tmux select-pane -t 3
tmux send-keys "conda activate reorder_nat" C-m
tmux send-keys "bash call_scripts/tool/look_exist_best_5.sh \
    -e J-2-3-1-H12-UR40T \
    -e Z-6-3-1-N-UF30T \
    -e b-6-3-1-N-UF30T \
    -e 2-2-3-1-H1-UR40M \
    -e 2-2-3-1-H2-UR40M \
    -e 2-2-3-1-H3-UR40M \
    -e 2-2-3-1-H4-UR40M \
    -e O-2-3-1-H12-UR40M \
    -e N-2-3-1-H12-UR40M \
    -e a-2-3-1-H12-UR40M \
    -e b-2-3-1-H12-UR40M \
    -e Y-2-3-1-N-UR40T \
    --sleep 120" C-m

# Attach to the tmux session to view the windows
tmux attach-session -t mysession
