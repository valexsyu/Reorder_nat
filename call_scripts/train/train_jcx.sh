source $HOME/.bashrc 
conda activate base
function pair_experiment() { 
    relay_step=70000
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    fi

    if [ "$cur_last" -lt $relay_step ]; then    
        bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens 1536 \
                                        --lm-start-step 75000 \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --hydra \
                                        -g 2 --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        

    if [ "$cur_last" -ge $relay_step ]; then
        if [ -e checkpoints/$1/top5_$relay_step/checkpoint_last.pt ] && \
        [ $(ls checkpoints/$1/top5_$relay_step/checkpoint.best_bleu_* 2>/dev/null \
                | grep -c "^checkpoints/$1/top5_$relay_step/checkpoint.best_bleu_.*") -eq 5 ]; then  
            echo "$1 6 checkpoint in top5_$relay_step"
        else
            echo "save top 5 before $relay_step"
            mkdir checkpoints/$1/top5_$relay_step
            "put -r $local_path" | sftp valex1377@xdata1.twcc.ai:$destination_path
            cp checkpoints/$1/checkpoint.best_bleu_* checkpoints/$1/top5_$relay_step
            cp checkpoints/$1/checkpoint_last.pt checkpoints/$1/top5_$relay_step
        fi
        for experiment in $2 $3 $4; do
            if [ -e checkpoints/$experiment/checkpoint_last.pt ] && \
            [ $(ls checkpoints/$experiment/checkpoint.best_bleu_* 2>/dev/null | grep -c "^checkpoints/$experiment/checkpoint.best_bleu_.*") -eq 5 ]; then    
                echo "$experiment 6 checkpoint files exist"
            else 
                mkdir checkpoints/$experiment/
                cp checkpoints/$1/top5_$relay_step/checkpoint.best_bleu_* checkpoints/$experiment/
                cp checkpoints/$1/top5_$relay_step/checkpoint_last.pt checkpoints/$experiment/     
            fi     
        done
    fi
    
    for experiment in $1 $2 $3 $4; do
        bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens 1536 \
                                        --lm-start-step 75000 \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update 100000 \
                                        --hydra \
                                        -g 2 --fp16        
    done                                                                                                                                                

}

pair_experiment 2-2-1-1-H12-UR40T 2-2-1-1-N-UR40T 2-2-1-1-H4-UR40T 2-2-1-1-H7-UR40T
pair_experiment 2-2-3-1-H12-UR40T 2-2-3-1-N-UR40T 2-2-3-1-H4-UR40T 2-2-3-1-H7-UR40T





