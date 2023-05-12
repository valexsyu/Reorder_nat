source $HOME/.bashrc 
conda activate base

function pair_experiment() { 
    bash call_scripts/train_nat.sh -e $1 \
                                    --save-interval-updates 70000 --max-tokens 4096 \
                                    --lm-start-step 75000 \
                                    --task translation_ctcpmlm \
                                    --arch nat_pretrained_model \
                                    --criterion nat_ctc_loss \
                                    --has-eos --max-update 70000 \
                                    --hydra \
                                    -g 2 --fp16       

    for experiment in $2 $3 $4; do
        mkdir checkpoints/$experiment/
        cp checkpoints/$1/checkpoint.best_bleu_* checkpoints/$experiment/
        cp checkpoints/$1/checkpoint_last.pt checkpoints/$experiment/
    done
    
    for experiment in $1 $2 $3 $4; do
        bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates 70000 --max-tokens 4096 \
                                        --lm-start-step 75000 \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update 100000 \
                                        --hydra \
                                        -g 2 --fp16        
    done                                                                                                                                                

}

pair_experiment 1-1-1-1-H12-UF20M 1-1-1-1-N-UF20M 1-1-1-1-H10-UF20M 1-1-1-1-H4-UF20M
pair_experiment 1-1-3-1-H12-UF20M 1-1-3-1-N-UF20M 1-1-3-1-H10-UF20M 1-1-3-1-H4-UF20M





