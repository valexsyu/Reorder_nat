experiment_1=2-2-1-1-H12-UR40M-comp
experiment_2=2-2-1-1-N-UR40M-comp 

function pair_experiment() { 
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/train_nat.sh -e $1 \
                        --twcc --fp16 --save-interval-updates 70000 \
                        --max-update 70000 --max-tokens 3072 \
                        --no-atten-mask \
                        -g 4 --fp16   

    for experiment in $2 $3 $4; do
        if [ -e checkpoints/$experiment/checkpoint_last.pt ] && \
        [ $(ls checkpoints/$experiment/checkpoint.best_bleu_* 2>/dev/null | grep -c "^checkpoints/$experiment/checkpoint.best_bleu_.*") -eq 5 ]; then    
            echo "All 6 checkpoint files exist"
        else 
            mkdir checkpoints/$experiment/
            mkdir checkpoints/$experiment/top5_before_70000
            cp checkpoints/$1/checkpoint.best_bleu_* checkpoints/$experiment/top5_before_70000
            cp checkpoints/$1/checkpoint.best_bleu_* checkpoints/$experiment/
            cp checkpoints/$1/checkpoint_last.pt checkpoints/$experiment/     
        fi     
    done
    
    for experiment in $1; do
        CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/train_nat.sh -e $experiment \
                        --twcc --fp16 --save-interval-updates 70000 \
                        --max-update 100000 --max-tokens 3072 \
                        --no-atten-mask \
                        -g 4 --fp16   
    done                                                                                                                                                

}

pair_experiment $experiment_1 $experiment_2             

CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/generate_nat.sh -b 1 --twcc --data-subset test --ck-types top --avg-speed 1 --no-atten-mask \
                    -e $experiment_1 \