source $HOME/.bashrc 
conda activate base


bash call_scripts/tool/line-by-line.sh --data-subset valid --ck-type top --b 50 --force \
-e 2-2-1-1-H12-UD33M \
-e 2-2-1-1-H12-UD30M \
-e 2-2-1-1-H12-UD25M \

# bash call_scripts/tool/line-by-line.sh --data-subset test --ck-type top --b 50 \
# -e 2-2-1-1-H12-UD33M \
# -e 2-2-1-1-H12-UD30M \
# -e 2-2-1-1-H12-UD25M \
