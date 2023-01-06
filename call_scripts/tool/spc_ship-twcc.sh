#!/bin/sh

# https://www.runoob.com/linux/linux-comm-scp.html


experiments=("2-2-2-1-H12-UF20M" "2-2-2-1-N-UF20M" "2-2-4-1-H12-UF20M" "2-2-4-1-N-UF20M")
# scp -P 59609 -r valex1377@203.145.216.187:/work/valex1377/CTC_PLM/Reorder_nat/checkpoints/$experiment checkpoints/

# exit 1
for experiment in "${experiments[@]}"; do
    scp -P 59609 -r valex1377@203.145.216.187:/work/valex1377/CTC_PLM/Reorder_nat/checkpoints/$experiment checkpoints/
    echo "$experiment processing==========================="
done