source $HOME/.bashrc 
conda activate base

# bash call_scripts/train_nat.sh -e  a-2-1-1-H12-UR40M --fp16 -g 1 --save-interval-updates 10000 --max-update 100000 \
#                                                      --lm-start-step 75000 --max-tokens 2048 -b 65536 --no-atten-mask --has-eos

function pair_experiment() {
experiment_1=$1
experiment_2=$2

hrun -s -N s03 -GG -c 12 -m 30 -t 3-0 bash call_scripts/train_nat.sh \
        -e  $experiment_1 --fp16 -g 2 --save-interval-updates 70000 --max-update 70000 \
        --lm-start-step 75000 --max-tokens 1024 -b 12288 --no-atten-mask --has-eos
mkdir checkpoints/$experiment_1/top5_70000steps    
cp checkpoints/$experiment_1/checkpoint.best_bleu_*  checkpoints/$experiment_1/top5_70000steps
mkdir checkpoints/$experiment_2/
cp checkpoints/$experiment_1/top5_70000steps/* checkpoints/$experiment_2/
cp checkpoints/$experiment_1/checkpoint_last.pt checkpoints/$experiment_2/
hrun -s -N s03 -GG -c 12 -m 30 -t 3-0 bash call_scripts/train_nat.sh \
        -e  $experiment_1 --fp16 -g 2 --save-interval-updates 70000 --max-update 100000 \
        --lm-start-step 75000 --max-tokens 1024 -b 12288 --no-atten-mask --has-eos
hrun -s -N s03 -GG -c 12 -m 30 -t 3-0 bash call_scripts/train_nat.sh \
        -e  $experiment_2 --fp16 -g 2 --save-interval-updates 70000 --max-update 100000 \
        --lm-start-step 75000 --max-tokens 1024 -b 12288 --no-atten-mask --has-eos    
}

pair_experiment 2-2-3-1-H12-UF20T-eos 2-2-3-1-H12-UF20T-eos







hrun -s -N s03 -GG -c 12 -m 30 -t 3-0 bash call_scripts/train_nat.sh \
        -e  2-2-4-1-H12-UF20T-eos --fp16 -g 2 --save-interval-updates 70000 --max-update 100000 \
        --lm-start-step 75000 --max-tokens 1024 -b 12288 --no-atten-mask --has-eos        
                                                    