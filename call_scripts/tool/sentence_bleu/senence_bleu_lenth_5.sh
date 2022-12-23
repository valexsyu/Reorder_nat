source $HOME/.bashrc 
conda activate base


# bash call_scripts/tool/line-by-line.sh --data-subset valid --ck-type top --b 50 --force \
# -e 2-2-1-1-H12-UR22M \
# -e 2-2-1-1-H12-UR20M \
# -e 2-2-1-1-H12-UR15M \

# bash call_scripts/tool/line-by-line.sh --data-subset test --ck-type top --b 50 \
# -e 2-2-1-1-H12-UR22M \
# -e 2-2-1-1-H12-UR20M \
# -e 2-2-1-1-H12-UR15M \

hrun -s -N s01 -G -c 20 -m 20 -t 3-0 bash call_scripts/tool/line-by-line.sh --data-subset train --b 30 \
-e 2-2-1-1-H12-UR25M \
e 2-2-1-1-H12-UR22M \
