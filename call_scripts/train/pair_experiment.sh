#!/bin/bash


# ctcpmlm $experiment $relay_step $MAX_TOKENS $LM_START_STEP $WARMUP_UPDATES $GPU_NUM
function ctcpmlm(){
    bash call_scripts/train_nat.sh -e $1 \
                                    --save-interval-updates $2 --max-tokens $3 \
                                    --lm-start-step $4 \
                                    --task translation_ctcpmlm \
                                    --arch nat_pretrained_model \
                                    --criterion nat_ctc_loss \
                                    --has-eos --max-update $7 \
                                    --warmup-updates $5 \
                                    --hydra \
                                    -b $8 \
                                    -g $6 --fp16       
}


function ctcpmlm_rate_avg(){
    bash call_scripts/train_nat.sh -e $1 \
                                    --save-interval-updates $2 --max-tokens $3 \
                                    --lm-start-step $4 \
                                    --arch ctcpmlm_rate_selection \
                                    --task translation_ctcpmlm \
                                    --criterion nat_ctc_avg_rate_loss \
                                    --has-eos --max-update $7 \
                                    --warmup-updates $5 \
                                    --hydra \
                                    -b $8 \
                                    -g $6 --fp16       
}

# cur_last=$(current_last_step $1)
function current_last_step(){
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        # echo "===========Loading $1  step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        # echo "Currect step: $cur_last"
    else
        # echo "===========No checkpoint_last.pt set cur_last=0=============="
        cur_last=0       
    fi
    # Return the value of cur_last
    echo "$cur_last"
}



# record_top5 $cur_last $relay_step $1 $2 $3 $4
function record_top5(){
    if [ "$1" -ge $2 ]; then
        if [ -e checkpoints/$3/top5_$2/checkpoint_last.pt ] && \
        [ $(ls checkpoints/$3/top5_$2/checkpoint.best_bleu_* 2>/dev/null \
                | grep -c "^checkpoints/$3/top5_$2/checkpoint.best_bleu_.*") -eq 5 ]; then  
            echo "$3 6 checkpoint in top5_$2"
        else
            echo "save top 5 before $2"
            mkdir checkpoints/$3/top5_$2
            cp checkpoints/$3/checkpoint.best_bleu_* checkpoints/$3/top5_$2
            cp checkpoints/$3/checkpoint_last.pt checkpoints/$3/top5_$2
        fi
        for experiment in $4 $5 $6; do
            if [ -e checkpoints/$experiment/checkpoint_last.pt ] && \
            [ $(ls checkpoints/$experiment/checkpoint.best_bleu_* 2>/dev/null | grep -c "^checkpoints/$experiment/checkpoint.best_bleu_.*") -eq 5 ]; then    
                echo "$experiment 6 checkpoint files exist"
            else 
                mkdir checkpoints/$experiment/
                cp checkpoints/$3/top5_$2/checkpoint.best_bleu_* checkpoints/$experiment/
                cp checkpoints/$3/top5_$2/checkpoint_last.pt checkpoints/$experiment/     
            fi     
        done
    fi    

}


#==================iwslt14de-en=====================

function pair_experiment(){
    relay_step=30000
    LM_START_STEP=30000
    MAX_TOKENS=4096
    GPU_NUM=1
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=50000

    cur_last=$(current_last_step $1)

    if [ "$cur_last" -lt $relay_step ]; then   
        ctcpmlm $1 $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM $relay_step $BATCH_SIZE
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)

    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        ctcpmlm $experiment $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM  $MAX_UPDATE $BATCH_SIZE
    done                                                                                                                                                

}


function pair_experiment_iwslt14_1_1536_rate_avg_33k(){
    relay_step=25000
    LM_START_STEP=25000
    MAX_TOKENS=1536
    GPU_NUM=1
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=33333

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt "$relay_step" ]; then   
        ctcpmlm_rate_avg $1 $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM $relay_step $BATCH_SIZE
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        ctcpmlm_rate_avg $experiment $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM  $MAX_UPDATE $BATCH_SIZE
    done                                                                                                                                                

}
function pair_experiment_iwslt14_2_1536_rate_avg_33k(){
    relay_step=25000
    LM_START_STEP=25000
    MAX_TOKENS=1536
    GPU_NUM=2
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=33333

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt "$relay_step" ]; then   
        ctcpmlm_rate_avg $1 $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM $relay_step $BATCH_SIZE
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        ctcpmlm_rate_avg $experiment $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM  $MAX_UPDATE $BATCH_SIZE
    done                                                                                                                                                

}

function pair_experiment_iwslt14() { 
    relay_step=30000
    LM_START_STEP=30000
    MAX_TOKENS=768
    GPU_NUM=1
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
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
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        --hydra \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
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
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        --hydra \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                

}
function pair_experiment_iwslt14_3080x2_1024() { 
    relay_step=30000
    LM_START_STEP=30000
    MAX_TOKENS=1024
    GPU_NUM=2
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
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
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        --hydra \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
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
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        --hydra \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                

}
function pair_experiment_iwslt14_3080x2_1536_100k() { 
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=1536
    GPU_NUM=2
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000
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
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        --hydra \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
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
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        --hydra \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                

}
function pair_experiment_iwslt14_3080x1_1536_100k() { 
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=1536
    GPU_NUM=1
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000
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
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        --hydra \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
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
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        --hydra \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                

}
function pair_experiment_iwslt14_2_2048_50k() { 
    relay_step=30000
    LM_START_STEP=30000
    MAX_TOKENS=2048
    GPU_NUM=2
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
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
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        --hydra \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
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
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        --hydra \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                

}
function pair_experiment_iwslt14_4_1024_50k() { 
    relay_step=30000
    LM_START_STEP=30000
    MAX_TOKENS=1024
    GPU_NUM=4
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
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
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        --hydra \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
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
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        --hydra \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                

}

function pair_experiment_iwslt14_3080x1_768_50k() { 
    relay_step=30000
    LM_START_STEP=30000
    MAX_TOKENS=768
    GPU_NUM=1
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=50000
    cur_last=$(current_last_step $1)

    if [ "$cur_last" -lt $relay_step ]; then   
        ctcpmlm $1 $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM $relay_step $BATCH_SIZE
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        ctcpmlm $experiment $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM  $MAX_UPDATE $BATCH_SIZE
    done                                                                                                                                                  

}
function pair_experiment_iwslt14_3080x2_768_50k() { 
    relay_step=30000
    LM_START_STEP=30000
    MAX_TOKENS=768
    GPU_NUM=2
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
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
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        --hydra \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
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
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        --hydra \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                

}
##
function pair_experiment_iwslt14_1_2048_50k(){
    relay_step=30000
    LM_START_STEP=30000
    MAX_TOKENS=2048
    GPU_NUM=1
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=50000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt "$relay_step" ]; then   
        ctcpmlm $1 $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM $relay_step $BATCH_SIZE
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        ctcpmlm $experiment $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM  $MAX_UPDATE $BATCH_SIZE
    done                                                                                                                                                

}
##
function pair_experiment_iwslt14_2_1024_50k(){
    relay_step=30000
    LM_START_STEP=30000
    MAX_TOKENS=1024
    GPU_NUM=2
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=50000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt "$relay_step" ]; then   
        ctcpmlm $1 $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM $relay_step $BATCH_SIZE
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        ctcpmlm $experiment $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM  $MAX_UPDATE $BATCH_SIZE
    done                                                                                                                                                

}
##
function pair_experiment_iwslt14_1_1024_50k(){
    relay_step=30000
    LM_START_STEP=30000
    MAX_TOKENS=1024
    GPU_NUM=1
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=50000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt "$relay_step" ]; then   
        ctcpmlm $1 $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM $relay_step $BATCH_SIZE
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        ctcpmlm $experiment $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM  $MAX_UPDATE $BATCH_SIZE
    done                                                                                                                                                

}
function pair_experiment_iwslt14_2_3072_50k(){
    relay_step=30000
    LM_START_STEP=30000
    MAX_TOKENS=3072
    GPU_NUM=2
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=50000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt "$relay_step" ]; then   
        ctcpmlm $1 $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM $relay_step $BATCH_SIZE
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        ctcpmlm $experiment $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM  $MAX_UPDATE $BATCH_SIZE
    done                                                                                                                                                

}

function pair_experiment_iwslt14_2_2048_100k(){
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=2048
    GPU_NUM=2
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt "$relay_step" ]; then   
        ctcpmlm $1 $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM $relay_step $BATCH_SIZE
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        ctcpmlm $experiment $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM  $MAX_UPDATE $BATCH_SIZE
    done                                                                                                                                                

}


function pair_experiment_iwslt14_1_4096_100k(){
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=4096
    GPU_NUM=1
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt "$relay_step" ]; then   
        ctcpmlm $1 $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM $relay_step $BATCH_SIZE
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        ctcpmlm $experiment $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM  $MAX_UPDATE $BATCH_SIZE
    done                                                                                                                                                

}
function pair_experiment_iwslt14_2_4096_100k(){
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=4096
    GPU_NUM=2
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt "$relay_step" ]; then   
        ctcpmlm $1 $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM $relay_step $BATCH_SIZE
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        ctcpmlm $experiment $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM  $MAX_UPDATE $BATCH_SIZE
    done                                                                                                                                                

}

function pair_experiment_iwslt14_1_2048_100k(){
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=2048
    GPU_NUM=1
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt "$relay_step" ]; then   
        ctcpmlm $1 $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM $relay_step $BATCH_SIZE
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        ctcpmlm $experiment $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM  $MAX_UPDATE $BATCH_SIZE
    done                                                                                                                                                

}
function pair_experiment_iwslt14_3_2048_50k(){
    relay_step=30000
    LM_START_STEP=30000
    MAX_TOKENS=2048
    GPU_NUM=3
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=50000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt "$relay_step" ]; then   
        ctcpmlm $1 $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM $relay_step $BATCH_SIZE
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        ctcpmlm $experiment $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM  $MAX_UPDATE $BATCH_SIZE
    done                                                                                                                                            

}
function pair_experiment_iwslt14_4_1536_50k(){
    relay_step=30000
    LM_START_STEP=30000
    MAX_TOKENS=1536
    GPU_NUM=4
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=50000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt "$relay_step" ]; then   
        ctcpmlm $1 $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM $relay_step $BATCH_SIZE
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        ctcpmlm $experiment $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM  $MAX_UPDATE $BATCH_SIZE
    done                                                                                                                                            

}
function pair_experiment_iwslt14_4_1536_100k(){
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=1536
    GPU_NUM=4
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt "$relay_step" ]; then   
        ctcpmlm $1 $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM $relay_step $BATCH_SIZE
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        ctcpmlm $experiment $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM  $MAX_UPDATE $BATCH_SIZE
    done
}
function pair_experiment_iwslt14_2_3072_100k(){
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=3072
    GPU_NUM=2
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt "$relay_step" ]; then   
        ctcpmlm $1 $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM $relay_step $BATCH_SIZE
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        ctcpmlm $experiment $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM  $MAX_UPDATE $BATCH_SIZE
    done                                                                                                                                                

}
function pair_experiment_iwslt14_3_2048_100k(){
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=2048
    GPU_NUM=3
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt "$relay_step" ]; then   
        ctcpmlm $1 $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM $relay_step $BATCH_SIZE
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        ctcpmlm $experiment $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM  $MAX_UPDATE $BATCH_SIZE
    done                                                                                                                                                

}
function pair_experiment_iwslt14_3_2048_50k_debug(){
    relay_step=30000
    LM_START_STEP=30000
    MAX_TOKENS=2048
    GPU_NUM=3
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=50000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt $relay_step ]; then    
        bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --debug \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                         
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --debug \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                 

}



#==================wmt14de-en=====================
function pair_experiment_wmt14_2_2048_100k() { 
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=2048
    GPU_NUM=2
    BATCH_SIZE=65536
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000
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
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
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
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                

}

function pair_experiment_wmt14_1_2048_100k() { 
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=2048
    GPU_NUM=1
    BATCH_SIZE=65536
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000
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
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
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
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                

}

function pair_experiment_wmt14_4_2048_100k() { 
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=2048
    GPU_NUM=4
    BATCH_SIZE=65536
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000
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
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
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
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                

}

function pair_experiment_wmt14_3080x1() { 
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=768
    GPU_NUM=1
    BATCH_SIZE=65536
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000
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
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
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
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                

}
function pair_experiment_wmt14_2_768_100k() { 
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=768
    GPU_NUM=2
    BATCH_SIZE=65536
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000
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
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
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
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                

}

#==================wmt14ro-en=====================
function pair_experiment_wmt16roen_2_4096_100k(){
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=4096
    GPU_NUM=2
    BATCH_SIZE=65536
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt "$relay_step" ]; then   
        ctcpmlm $1 $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM $relay_step $BATCH_SIZE
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        ctcpmlm $experiment $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM  $MAX_UPDATE $BATCH_SIZE
    done                                                                                                                                                

}
function pair_experiment_wmt16roen_2_2730_100k(){
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=2730
    GPU_NUM=2
    BATCH_SIZE=65520
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt "$relay_step" ]; then   
        ctcpmlm $1 $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM $relay_step $BATCH_SIZE
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        ctcpmlm $experiment $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM  $MAX_UPDATE $BATCH_SIZE
    done                                                                                                                                                

}
function pair_experiment_wmt16roen_4_4096_rate_avg_33k(){
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=4096
    GPU_NUM=4
    BATCH_SIZE=65536
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt "$relay_step" ]; then   
        ctcpmlm_rate_avg $1 $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM $relay_step $BATCH_SIZE
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        ctcpmlm_rate_avg $experiment $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM  $MAX_UPDATE $BATCH_SIZE
    done                                                                                                                                                

}
function pair_experiment_wmt16roen_2_2730_rate_avg_33k(){
    relay_step=25000
    LM_START_STEP=25000
    MAX_TOKENS=2730
    GPU_NUM=2
    BATCH_SIZE=65520
    WARMUP_UPDATES=10000
    MAX_UPDATE=33333

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt "$relay_step" ]; then   
        ctcpmlm_rate_avg $1 $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM $relay_step $BATCH_SIZE
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        ctcpmlm_rate_avg $experiment $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM  $MAX_UPDATE $BATCH_SIZE
    done                                                                                                                                                

}
function pair_experiment_wmt16roen_2_2048_rate_avg_33k(){
    relay_step=25000
    LM_START_STEP=25000
    MAX_TOKENS=2048
    GPU_NUM=2
    BATCH_SIZE=65536
    WARMUP_UPDATES=10000
    MAX_UPDATE=33333

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt "$relay_step" ]; then   
        ctcpmlm_rate_avg $1 $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM $relay_step $BATCH_SIZE
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        ctcpmlm_rate_avg $experiment $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM  $MAX_UPDATE $BATCH_SIZE
    done                                                                                                                                                

}



#============distill_layer 12+1 experiment=======


# record_top5 $cur_last $relay_step $1 $2 $3 $4
function record_top5_11(){
    if [ "$1" -ge $2 ]; then
        if [ -e checkpoints/$3/top5_$2/checkpoint_last.pt ] && \
        [ $(ls checkpoints/$3/top5_$2/checkpoint.best_bleu_* 2>/dev/null \
                | grep -c "^checkpoints/$3/top5_$2/checkpoint.best_bleu_.*") -eq 5 ]; then  
            echo "$3 6 checkpoint in top5_$2"
        else
            echo "save top 5 before $2"
            mkdir checkpoints/$3/top5_$2
            cp checkpoints/$3/checkpoint.best_bleu_* checkpoints/$3/top5_$2
            cp checkpoints/$3/checkpoint_last.pt checkpoints/$3/top5_$2
        fi
        for experiment in $4 $5 $6 $7 $8 $9 $10 $11 $12 $13 ; do
            if [ -e checkpoints/$experiment/checkpoint_last.pt ] && \
            [ $(ls checkpoints/$experiment/checkpoint.best_bleu_* 2>/dev/null | grep -c "^checkpoints/$experiment/checkpoint.best_bleu_.*") -eq 5 ]; then    
                echo "$experiment 6 checkpoint files exist"
            else 
                mkdir checkpoints/$experiment/
                cp checkpoints/$3/top5_$2/checkpoint.best_bleu_* checkpoints/$experiment/
                cp checkpoints/$3/top5_$2/checkpoint_last.pt checkpoints/$experiment/     
            fi     
        done
    fi    

}
function pair_experiment_iwslt14_1_2048_50k_11() { 
    relay_step=30000
    LM_START_STEP=30000
    MAX_TOKENS=2048
    GPU_NUM=1
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=50000
    cur_last=$(current_last_step $1)

    if [ "$cur_last" -lt $relay_step ]; then   
        ctcpmlm $1 $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM $relay_step $BATCH_SIZE
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)

    record_top5_11 $cur_last $relay_step $1 $2 $3 $4 $5 $6 $7 $8 $9 $10 $11

    
    for experiment in $1 $2 $3 $4 $5 $6 $7 $8 $9 $10 $11 ; do 
        ctcpmlm $experiment $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM $MAX_UPDATE $BATCH_SIZE
    done     
}





#=====================TWCC==============================
function pair_experiment_wmt14_2_3276_100k_twcc() { 
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=3276
    GPU_NUM=2
    BATCH_SIZE=65520
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
    fi

    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
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
        CUDA_VISIBLE_DEVICES=0,1 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                

}
function pair_experiment_wmt16_8_2048_30k_twcc() { 
    relay_step=19000
    LM_START_STEP=20000
    MAX_TOKENS=2048
    GPU_NUM=8
    BATCH_SIZE=65536
    WARMUP_UPDATES=3000
    MAX_UPDATE=30000

    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
    fi

    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
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
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                   

}
function pair_experiment_wmt14_4_3276_100k_twcc() { 
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=3276
    GPU_NUM=4
    BATCH_SIZE=65520
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
    fi

    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
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
        CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                

}
function pair_experiment_wmt14_8_2048_100k_twcc() { 
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=2048
    GPU_NUM=8
    BATCH_SIZE=65536
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
    fi

    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4
    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                

}
function pair_experiment_wmt14_8_2730_100k_twcc() { 
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=2730
    GPU_NUM=8
    BATCH_SIZE=65520
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
    fi

    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4
    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                

}
function pair_experiment_wmt14_8_4095_100k_twcc() { 
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=4095
    GPU_NUM=8
    BATCH_SIZE=65520
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
    fi

    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4
    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                

}
function pair_experiment_wmt14_8_1638_rate_avg_33k_twcc(){
    relay_step=25000
    LM_START_STEP=25000
    MAX_TOKENS=1638
    GPU_NUM=8
    BATCH_SIZE=65536
    WARMUP_UPDATES=10000
    MAX_UPDATE=33333

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_avg_rate_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                         
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_avg_rate_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                 

}
function pair_experiment_wmt14_8_1638_rate_avg_33k_warm33_twcc(){
    relay_step=25000
    LM_START_STEP=25000
    MAX_TOKENS=1638
    GPU_NUM=8
    BATCH_SIZE=65536
    WARMUP_UPDATES=3333
    MAX_UPDATE=33333

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_avg_rate_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                         
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_avg_rate_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                 

}
function pair_experiment_wmt16_8_2048_rate_avg_33k_twcc(){
    relay_step=25000
    LM_START_STEP=25000
    MAX_TOKENS=2048
    GPU_NUM=8
    BATCH_SIZE=65536
    WARMUP_UPDATES=3000
    MAX_UPDATE=33333

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    
    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_avg_rate_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                         
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_avg_rate_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                        

}




function pair_experiment_iwslt14_2_3072_50k_twcc(){
    relay_step=30000
    LM_START_STEP=30000
    MAX_TOKENS=3072
    GPU_NUM=2
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=50000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                         
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                 

}
function pair_experiment_iwslt14_2_2048_50k_twcc(){
    relay_step=30000
    LM_START_STEP=30000
    MAX_TOKENS=2048
    GPU_NUM=2
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=50000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                         
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                 

}
function pair_experiment_iwslt14_2_1024_50k_twcc(){
    relay_step=30000
    LM_START_STEP=30000
    MAX_TOKENS=1024
    GPU_NUM=2
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=50000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                         
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                 

}
function pair_experiment_iwslt14_2_2048_100k_twcc(){
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=2048
    GPU_NUM=2
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                         
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                 

}
function pair_experiment_iwslt14_2_3072_100k_twcc(){
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=3072
    GPU_NUM=2
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                         
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                 

}
function pair_experiment_iwslt14_2_4096_100k_twcc(){
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=4096
    GPU_NUM=2
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                         
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                 

}
function pair_experiment_iwslt14_4_2048_100k_twcc(){
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=2048
    GPU_NUM=4
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                         
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                 

}
function pair_experiment_iwslt14_4_1536_rate_avg_33k_twcc(){
    relay_step=25000
    LM_START_STEP=25000
    MAX_TOKENS=1536
    GPU_NUM=4
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=33333

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_avg_rate_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                         
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_avg_rate_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                 

}
function pair_experiment_iwslt14_8_1536_rate_avg_33k_warm33_twcc(){
    relay_step=25000
    LM_START_STEP=25000
    MAX_TOKENS=1536
    GPU_NUM=8
    BATCH_SIZE=12288
    WARMUP_UPDATES=3333
    MAX_UPDATE=33333

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_avg_rate_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                         
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_avg_rate_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                 

}
function pair_experiment_iwslt14_4_1536_rate_avg_twcc(){
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=1536
    GPU_NUM=4
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_avg_rate_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                         
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_avg_rate_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                 

}
function pair_experiment_iwslt14_4_2048_rate_avg_1_twcc(){
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=2048
    GPU_NUM=4
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_avg_rate_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        --rate-list 1 \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                         
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_avg_rate_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        --rate-list 1 \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                 

}
function pair_experiment_iwslt14_4_1536_rate_avg_1_twcc(){
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=1536
    GPU_NUM=4
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_avg_rate_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        --rate-list 1 \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                         
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_avg_rate_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        --rate-list 1 \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                 

}
function pair_experiment_iwslt14_8_1536_rate_avg_1_20k_twcc(){
    relay_step=15000
    LM_START_STEP=15000
    MAX_TOKENS=1536
    GPU_NUM=8
    BATCH_SIZE=12288
    WARMUP_UPDATES=5000
    MAX_UPDATE=20000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_avg_rate_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        --rate-list 1 \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                         
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_avg_rate_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        --rate-list 1 \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                 

}
#================local==================

function pair_experiment_iwslt14_3080x1_768_50k_loacl() { 
    relay_step=30000
    LM_START_STEP=30000
    MAX_TOKENS=768
    GPU_NUM=1
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
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
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        --hydra \
                                        --local \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
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
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        --hydra \
                                        --local \
                                        -g $GPU_NUM --fp16        
    done     
}



function pair_experiment_wmt14_8_2730_100k_twcc() { 
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=2730
    GPU_NUM=8
    BATCH_SIZE=65520
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
    fi

    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4
    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                

}
function pair_experiment_wmt14_8_2048_100k_twcc_reset_meter() { 
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=2048
    GPU_NUM=8
    BATCH_SIZE=65536
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
    fi

    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        --reset-meter \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4
    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                

}
function pair_experiment_wmt14_8_4095_100k_twcc_reset_meter() { 
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=4095
    GPU_NUM=8
    BATCH_SIZE=65520
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
    fi

    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        --reset-meter \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4
    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                

}
function pair_experiment_wmt16roen_2_4096_100k_reset_meter(){
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=4096
    GPU_NUM=2
    BATCH_SIZE=65536
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt $relay_step ]; then    
        bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --reset-meter \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                       
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        ctcpmlm $experiment $relay_step $MAX_TOKENS \
                $LM_START_STEP $WARMUP_UPDATES $GPU_NUM  $MAX_UPDATE $BATCH_SIZE
    done                                                                                                                                                

}



function pair_experiment_wmt16_8_4096_100k_twcc() { 
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=4096
    GPU_NUM=8
    BATCH_SIZE=65536
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000

    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
    fi

    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
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
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                   

}

function pair_experiment_wmt16_8_2730_100k_twcc() { 
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=2730
    GPU_NUM=8
    BATCH_SIZE=65536
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000

    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
    fi

    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    if [ -e checkpoints/$1/checkpoint_last.pt ]; then
        echo "===========Loading $1 checkpoint_last step=============="
        cur_last=$(python call_scripts/tool/load_checkpoint_step.py checkpoints/$1/ last \
                  | awk -F':' '/last/{gsub(/[^0-9]/, "", $3); print $3}')
        echo "Currect step: $cur_last"
    else
        cur_last=0        
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
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                   

}

function pair_experiment_wmt14_8_1638_rate_avg_33k_warmup3k_twcc(){
    relay_step=25000
    LM_START_STEP=25000
    MAX_TOKENS=1638
    GPU_NUM=8
    BATCH_SIZE=65536
    WARMUP_UPDATES=3333
    MAX_UPDATE=33333

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_avg_rate_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                         
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_avg_rate_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                 

}


function pair_experiment_wmt14_8_1638_rate_avg_33k_50k_twcc(){
    relay_step=25000
    LM_START_STEP=25000
    MAX_TOKENS=1638
    GPU_NUM=8
    BATCH_SIZE=65536
    WARMUP_UPDATES=10000
    MAX_UPDATE=50000



    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ;  copy done "    



    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_avg_rate_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                 

}

function pair_experiment_iwslt14_8_1536_rate_avg_33k_w1_twcc(){
    relay_step=25000
    LM_START_STEP=25000
    MAX_TOKENS=1536
    GPU_NUM=8
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=33333

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_avg_rate_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        --rate-weight-list 1 \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                         
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_avg_rate_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        --rate-weight-list 1 \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                 

}

function pair_experiment_iwslt14_8_1536_rate_avg_33k_w2_twcc(){
    relay_step=25000
    LM_START_STEP=25000
    MAX_TOKENS=1536
    GPU_NUM=8
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=33333

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_avg_rate_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        --rate-weight-list 2 \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                         
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_avg_rate_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        --rate-weight-list 2 \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                 

}

function pair_experiment_wmt14_8_1638_rate_avg_33k_100k_twcc(){
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=1638
    GPU_NUM=8
    BATCH_SIZE=65520
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_avg_rate_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                         
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_avg_rate_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                 

}
function pair_experiment_wmt14_8_2048_100k_300k_twcc() { 
    relay_step=200000
    LM_START_STEP=22500
    MAX_TOKENS=2048
    GPU_NUM=8
    BATCH_SIZE=65536
    WARMUP_UPDATES=10000
    MAX_UPDATE=300000
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "

    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4
    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                

}      
function pair_experiment_wmt14_8_2730_100k_300k_twcc() { 
    relay_step=200000
    LM_START_STEP=22500
    MAX_TOKENS=2730
    GPU_NUM=8
    BATCH_SIZE=65520
    WARMUP_UPDATES=10000
    MAX_UPDATE=300000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "

    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4
    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                

}        
function pair_experiment_wmt14_8_4095_100k_300k_twcc() { 
    relay_step=200000
    LM_START_STEP=22500
    MAX_TOKENS=4095
    GPU_NUM=8
    BATCH_SIZE=65520
    WARMUP_UPDATES=10000
    MAX_UPDATE=300000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "

    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                        
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4
    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                

}
function pair_experiment_wmt14_8_1638_rate_sel_33k_twcc(){
    relay_step=25000
    LM_START_STEP=25000
    MAX_TOKENS=1638
    GPU_NUM=8
    BATCH_SIZE=65520
    WARMUP_UPDATES=10000
    MAX_UPDATE=33333

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_sel_rate_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                         
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --arch ctcpmlm_rate_selection \
                                        --task translation_ctcpmlm \
                                        --criterion nat_ctc_sel_rate_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                 

}



function pair_experiment_iwslt14_8_1536_100k_debug_twcc(){
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=1536
    GPU_NUM=8
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --debug \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                         
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --debug \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                 

}

function pair_experiment_iwslt14_8_1536_100k_twcc(){
    relay_step=70000
    LM_START_STEP=75000
    MAX_TOKENS=1536
    GPU_NUM=8
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=100000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                         
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                 

}
function pair_experiment_iwslt14_8_1536_50k_twcc(){
    relay_step=30000
    LM_START_STEP=30000
    MAX_TOKENS=1536
    GPU_NUM=8
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=50000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                         
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                 

}
function pair_experiment_iwslt14_8_1536_50k_debug_twcc(){
    relay_step=30000
    LM_START_STEP=30000
    MAX_TOKENS=1536
    GPU_NUM=8
    BATCH_SIZE=12288
    WARMUP_UPDATES=10000
    MAX_UPDATE=50000

    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step1 "
    
    if [ "$cur_last" -lt $relay_step ]; then    
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $1 \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $relay_step \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --debug \
                                        --twcc \
                                        -g $GPU_NUM --fp16   
    else
        echo "$1 last step is ge $relay_step"
    fi                                         
    cur_last=$(current_last_step $1)
    echo "Currect step: $cur_last ; Now is Step2 "
    record_top5 $cur_last $relay_step $1 $2 $3 $4

    
    for experiment in $1 $2 $3 $4; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment \
                                        --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                        --lm-start-step $LM_START_STEP \
                                        --task translation_ctcpmlm \
                                        --arch nat_pretrained_model \
                                        --criterion nat_ctc_loss \
                                        --has-eos --max-update $MAX_UPDATE \
                                        --warmup-updates $WARMUP_UPDATES \
                                        -b $BATCH_SIZE \
                                        --hydra \
                                        --debug \
                                        --twcc \
                                        -g $GPU_NUM --fp16        
    done                                                                                                                                                 

}


function pair_experiment_wmt14_8_4096_QK50k_twcc() { 
    relay_step=10000
    LM_START_STEP=50000
    MAX_TOKENS=4096
    GPU_NUM=8
    BATCH_SIZE=65536
    WARMUP_UPDATES=10000
    MAX_UPDATE=50000


    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $1 \
                                    --save-interval-updates $relay_step --max-tokens $MAX_TOKENS \
                                    --lm-start-step $LM_START_STEP \
                                    --arch ctcpmlm_low_rate_finetune \
                                    --task translation_ctcpmlm \
                                    --criterion nat_ctc_loss \
                                    --has-eos --max-update $MAX_UPDATE \
                                    --warmup-updates $WARMUP_UPDATES \
                                    -b $BATCH_SIZE \
                                    --hydra \
                                    --twcc \
                                    --reset-optimizer \
                                    -g $GPU_NUM --fp16        
                                                                                                                                       

}
