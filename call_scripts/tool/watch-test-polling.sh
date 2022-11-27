function default_setting() {
    twcc=False
    sleep_time=3600
    
}


VALID_ARGS=$(getopt -o e: --long experiment:,twcc,sleep: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1;
fi

eval set -- "$VALID_ARGS"

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
      if [ "$twcc" = "True" ]
      then
         echo "Wait TWCC Resource"
         bash call_scripts/generate_nat.sh -e $experiment_id --no-atten-mask -b 1 --ck-types last-top --cpu --twcc > tmp_file
      else
         echo "Wait Battleship Resource"
         hrun -s -c 20 -m 40 bash call_scripts/generate_nat.sh -e $experiment_id --no-atten-mask -b 1 --ck-types last-top > tmp_file
      fi
      score=$(tail -1 tmp_file) 
      echo $dt ': ' $'\t' $score >> $CHECKPOINT/best_top5.test.record
      rm tmp_file
   done
   echo "Sleeping Now"
   sleep $sleep_time

done

