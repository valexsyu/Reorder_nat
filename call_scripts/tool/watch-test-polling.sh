function default_setting() {
    twcc=False
    sleep_time=3600
    arch=nat_pretrained_model
    criterion=nat_ctc_loss
    task=translation_ctcpmlm
    local=False
    bz=50
    
}


VALID_ARGS=$(getopt -o e: --long experiment:,twcc,sleep:,local,b \
                          --long task:,arch:,criterion: -- "$@")
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
    --b)
      bz="$2"
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
         echo "Folder is not exist"
         continue
      fi       
      dt=$(date)
      random_num=$RANDOM  # while runing 2 watch in the same time to avoid write the same temp file. 
      if [ "$local" = "True" ]; then  
        echo "Wait local Resource"   
        CUDA_VISIBLE_DEVICES=1 bash call_scripts/generate_nat.sh -e $experiment_id -b $bz --ck-types last-top --local \
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

