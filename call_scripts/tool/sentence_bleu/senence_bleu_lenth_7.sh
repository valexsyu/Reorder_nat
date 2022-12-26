source $HOME/.bashrc 
conda activate base


# bash call_scripts/tool/line-by-line.sh --data-subset valid --ck-type top --b 50 --force \
# -e 2-2-1-1-H12-UD50M \
# -e 2-2-1-1-H12-UD45M \
# -e 2-2-1-1-H12-UD40M \

# bash call_scripts/generate_nat.sh --data-subset valid --ck-type top --b 50 --load-exist-bleu \
# -e 2-2-1-1-H12-UD50M \
# -e 2-2-1-1-H12-UD45M \
# -e 2-2-1-1-H12-UD40M \
# -e 2-2-1-1-H12-UD33M \
# -e 2-2-1-1-H12-UD30M \
# -e 2-2-1-1-H12-UD25M \
# -e 2-2-1-1-H12-UD22M \
# -e 2-2-1-1-H12-UD20M \
# -e 2-2-1-1-H12-UD15M \
# -e 2-2-1-1-H12-UR50M \
# -e 2-2-1-1-H12-UR45M \
# -e 2-2-1-1-H12-UR40M \
# -e 2-2-1-1-H12-UR33M \
# -e 2-2-1-1-H12-UR30M \
# -e 2-2-1-1-H12-UR25M \
# -e 2-2-1-1-H12-UR22M \
# -e 2-2-1-1-H12-UR20M \
# -e 2-2-1-1-H12-UR15M \

# bash call_scripts/tool/line-by-line.sh --data-subset test --ck-type top --b 50 \
# -e 2-2-1-1-H12-UD50M \
# -e 2-2-1-1-H12-UD45M \
# -e 2-2-1-1-H12-UD40M \

hrun -s -N s05 -c 4 -m 20 -t 3-0 bash call_scripts/tool/line-by-line.sh --data-subset train --b 20 \
-e 2-2-1-1-H12-UR50M \