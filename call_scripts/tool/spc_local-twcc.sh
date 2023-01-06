#!/bin/sh
experiments=("2-2-1-1-H5-UF20M" "2-2-1-1-H1-UF20T" "2-2-1-1-H2-UF20T" "2-2-1-1-H3-UF20T" "2-2-1-1-H4-UF20T" "2-2-1-1-H5-UF20T")
# scp -P 59609 -r valex1377@203.145.216.187:/work/valex1377/CTC_PLM/Reorder_nat/checkpoints/$experiment checkpoints/

# exit 1
for experiment in "${experiments[@]}"; do
    scp -P 59609 -r valex1377@203.145.216.187:/work/valex1377/CTC_PLM/Reorder_nat/checkpoints/$experiment checkpoints/
done