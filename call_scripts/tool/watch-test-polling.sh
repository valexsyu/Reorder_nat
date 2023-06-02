#!/bin/bash

function default_setting() {
    twcc=False
    sleep_time=3600
    arch=nat_pretrained_model
    criterion=nat_ctc_loss
    task=translation_ctcpmlm
    local=False
    bz=50
    gpu_id=0
    
}


VALID_ARGS=$(getopt -o e:,b: --long experiment:,twcc,sleep:,local \
                          --long task:,arch:,criterion:,gpu_id: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1;
fi

eval set -- "$VALID_ARGS"
default_setting

while [ : ]; do
  case "$1" in 
    -e | --experiment)
        experiment_ids="$2"     
        echo "$experiment_ids" 
        exp_array+=("$experiment_ids")
        shift 2
        ;;
    -b)
      bz="$2"
      shift 2
      ;;       
    --gpu_id)
      gpu_id="$2"
      shift 2
      ;;               
    --arch)
      arch="$2"
      shift 2
      ;;        
    --task)
      task="$2"
      shift 2
      ;;
    --criterion)
      criterion="$2"
      shift 2
      ;;             
    --twcc)
      twcc=True
      shift 1
      ;;      
    --local)
      local="True"
      shift 1
      ;;              
    --sleep)
      sleep_time=$2
      shift 2
      ;;                
    --) shift; 
        break
  esac
done


while :
do

   for i in "${!exp_array[@]}"; do 
      experiment_id=${exp_array[$i]}
      CHECKPOINT=checkpoints/$experiment_id
      if [ ! -d "$CHECKPOINT" ]; then
         echo "Folder $CHECKPOINT is not exist"
         continue
      fi       
      
      ## To reduce generation time, 
      ## skip iterations where the generated output exceeds 100,000 tokens 
      ## or when the last generated value remains unchanged
      pervious_value=$(tail -1 $CHECKPOINT/best_top5.test.record | grep -oE "last:[0-9]+" | grep -oE "[0-9]+")
      if [ -n "$pervious_value" ] && [ "$pervious_value" -ge 100000 ]; then
          echo "The $CHECKPOINT : Last value is equal to or greater than 100,000"
          score_array+=$(tail -1 $CHECKPOINT/best_top5.test.record) 
          continue
      else
          echo "Last Value Loading....."
          last_value=$(python call_scripts/tool/load_checkpoint_step.py $CHECKPOINT last | grep -oE "last:[0-9]+" | grep -oE "[0-9]+")
          if [ "$last_value" = "$previous_value" ]; then
          echo "The $CHECKPOINT : last generated value remains unchanged"
              continue
          fi
      fi

      dt=$(date)
      random_num=$RANDOM  # while runing 2 watch in the same time to avoid write the same temp file. 

      if [ "$local" = "True" ]; then  
        echo "Wait local Resource"   
        CUDA_VISIBLE_DEVICES=$gpu_id bash call_scripts/generate_nat.sh -e $experiment_id -b $bz --ck-types last-top --local \
                                          --arch $arch --task $task --criterion $criterion > tmp_file_$random_num      
      else
        echo "Wait Battleship Resource"
        bash call_scripts/generate_nat.sh -e $experiment_id -b $bz --ck-types last-top \
                                          --arch $arch --task $task --criterion $criterion > tmp_file_$random_num
      fi
      score=$(tail -1 tmp_file_$random_num) 
      score_array+=("$experiment_id : $score")
      echo $dt ': ' $'\t' $score >> $CHECKPOINT/best_top5.test.record
      rm tmp_file_$random_num
   done
   date
   printf '%s\n' "${score_array[@]}"
   echo "Sleeping Now : $sleep_time s"
   score_array=()
   sleep $sleep_time

done

