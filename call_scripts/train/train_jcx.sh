function pair_experiment() { 
    relay_step=30000
    LM_START_STEP=30000
    max_tokens=2048
    GPU_NUM=4
    MAX_UPDATE=50000
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
    fi

    if [ "$cur_last" -lt $relay_step ]; then    
        bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $max_tokens \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --lm-start-step $LM_START_STEP \
                                        --hydra \
                                        -g $GPU_NUM --fp16   
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
                                        --save-interval-updates $relay_step --max-tokens $max_tokens \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --hydra \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                

}


#=========NEW==========
pair_experiment 2-2-3-1-N-UR40M 2-2-3-1-H12-UR40M
pair_experiment Y-2-3-1-N-UR40M Y-2-3-1-H12-UR40M
pair_experiment Z-2-3-1-N-UR40M Z-2-3-1-H12-UR40M
pair_experiment C-2-3-1-N-UR40M C-2-3-1-H12-UR40M
pair_experiment D-2-3-1-N-UR40M D-2-3-1-H12-UR40M






