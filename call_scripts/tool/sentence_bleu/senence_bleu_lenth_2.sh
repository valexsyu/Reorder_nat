source $HOME/.bashrc 
conda activate base


# bash call_scripts/tool/line-by-line.sh --data-subset valid --ck-type top --b 50 --force \
# -e 2-2-1-1-H12-UD22M \
# -e 2-2-1-1-H12-UD20M \
# -e 2-2-1-1-H12-UD15M \


# bash call_scripts/tool/line-by-line.sh --data-subset test --ck-type top --b 50 \
# -e 2-2-1-1-H12-UD22M \
# -e 2-2-1-1-H12-UD20M \
# -e 2-2-1-1-H12-UD15M \

hrun -s -N s04 -G -c 4 -m 20 -t 3-0 bash call_scripts/tool/line-by-line.sh --data-subset train --b 20 \
-e 2-2-1-1-H12-UR40M \

