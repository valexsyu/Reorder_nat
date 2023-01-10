function default_setting() {
    twcc=False
    sleep_time=2500
    port=51645
    ip=203.145.216.187
    
}


VALID_ARGS=$(getopt -o e: --long experiment:,twcc,sleep:,port:,ip: -- "$@")
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
         bash call_scripts/generate_nat.sh -e $experiment_id -b 50 --ck-types last-top --has-eos > tmp_file
      else
         echo "Wait Battleship Resource"
         hrun -s -c 4 -m 12 scp -P $port -r valex1377@$ip:/work/valex1377/CTC_PLM/Reorder_nat/checkpoints/$experiment_id/checkpoint_best_top5.pt checkpoints/$experiment_id/
         echo "top5.pt transmission finish"
         hrun -s -c 4 -m 12 scp -P $port -r valex1377@$ip:/work/valex1377/CTC_PLM/Reorder_nat/checkpoints/$experiment_id/checkpoint_last.pt checkpoints/$experiment_id/
         hrun -s -c 4 -m 12 scp -P $port -r valex1377@$ip:/work/valex1377/CTC_PLM/Reorder_nat/checkpoints/$experiment_id/checkpoint_last5.pt checkpoints/$experiment_id/
         echo "last.pt transmission finish"
         random_num=$RANDOM
         hrun -s -N s05 -c 8 -m 40 bash call_scripts/generate_nat.sh -e $experiment_id -b 50 --ck-types top-lastk --avg-ck-turnoff > tmp_twcc_file_$random_num
      fi
      score=$(tail -1 tmp_twcc_file_$random_num) 
      echo $dt ': ' $'\t' $score >> $CHECKPOINT/best_top5.test.record
      rm tmp_twcc_file_$random_num
   done
   date
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