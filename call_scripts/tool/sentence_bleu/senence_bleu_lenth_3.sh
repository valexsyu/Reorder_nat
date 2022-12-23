source $HOME/.bashrc 
conda activate base


# bash call_scripts/tool/line-by-line.sh --data-subset valid --ck-type top --b 50 --force \
# -e 2-2-1-1-H12-UR50M \
# -e 2-2-1-1-H12-UR45M \
# -e 2-2-1-1-H12-UR40M \

# bash call_scripts/tool/line-by-line.sh --data-subset test --ck-type top --b 50 \
# -e 2-2-1-1-H12-UR50M \
# -e 2-2-1-1-H12-UR45M \
# -e 2-2-1-1-H12-UR40M \

hrun -s -N s03 -G -c 20 -m 20 -t 3-0 bash call_scripts/tool/line-by-line.sh --data-subset train --b 25 \
-e 2-2-1-1-H12-UR33M \

