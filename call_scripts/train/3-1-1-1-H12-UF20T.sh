source $HOME/.bashrc 
conda activate base

function pair_experiment() {
    experiment_1=$1
    experiment_2=$2
    bash call_scripts/train_nat.sh \
            -e  $experiment_1 --fp16 -g 2 --save-interval-updates 70000 --max-update 70000 \
            --lm-start-step 75000 --max-tokens 2048 -b 12288 --no-atten-mask
    mkdir checkpoints/$experiment_1/top5_70000steps    
    cp checkpoints/$experiment_1/checkpoint.best_bleu_*  checkpoints/$experiment_1/top5_70000steps
    mkdir checkpoints/$experiment_2/
    cp checkpoints/$experiment_1/top5_70000steps/* checkpoints/$experiment_2/
    cp checkpoints/$experiment_1/checkpoint_last.pt checkpoints/$experiment_2/
    bash call_scripts/train_nat.sh \
            -e  $experiment_1 --fp16 -g 2 --save-interval-updates 70000 --max-update 100000 \
            --lm-start-step 75000 --max-tokens 2048 -b 12288 --no-atten-mask
}


pair_experiment 3-1-1-1-H12-UF20T 3-1-1-1-N-UF20T













