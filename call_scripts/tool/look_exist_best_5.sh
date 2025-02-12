#!/bin/sh
# e.g. bash call_scripts/tool/look_exist_best_5.sh -e K-2-1-1-H7-UR40M -e m-B-1-1-N-UR20M \
#                         -e m-B-1-1-N-UR20M-rate_select-divTGT-NEW-3 -e 2-2-3-1-N-UR20M-rate_select-divTGT-NEW-3 \
#                         -e m-B-1-1-N-UR20M-rate_predict_divTGT-NEW-detach --sleep 60

function default_setting() {
    twcc=False
    sleep_time=10
    port=51645
    ip=203.145.216.187
    arch=nat_pretrained_model
}



VALID_ARGS=$(getopt -o e: --long experiment:,twcc,sleep:,port:,ip:,arch: -- "$@")
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
    --twcc)
      twcc=True
      shift 1
      ;;        
    --port)
      port=$2
      shift 2
      ;;   
    --arch)
      arch="$2"
      shift 2
      ;;            
    --ip)
      ip=$2
      shift 2
      ;;         
    --sleep)
      sleep_time=$2
      shift 2
      ;;                
    --) shift; 
        break
  esac
done

num_sleep=1
while :
do


   echo "===================================================="
   for i in "${!exp_array[@]}"; do 
      experiment_id=${exp_array[$i]}
      CHECKPOINT=checkpoints/$experiment_id
      if [ ! -d "$CHECKPOINT" ]; then
         echo "$experiment_id Folder is not exist"
         continue
      fi       
      score=$(tail -1 $CHECKPOINT/best_top5.test.record) 
      echo "$experiment_id : $score "
   done
   echo "Sleeping Now : $sleep_time s "
   sleep $sleep_time

done


# experiments=("J-2-1-1-H12-UF20T" "J-2-1-1-H12-UF20M")
# # scp -P 59609 -r valex1377@203.145.216.187:/work/valex1377/CTC_PLM/Reorder_nat/checkpoints/$experiment checkpoints/

# # exit 1
# for experiment in "${experiments[@]}"; do
#     scp -P 59609 -r valex1377@203.145.216.187:/work/valex1377/CTC_PLM/Reorder_nat/checkpoints/$experiment checkpoints/
#     echo "$experiment processing==========================="
# done