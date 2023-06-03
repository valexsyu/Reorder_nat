# source $HOME/.bashrc 
# conda activate base

CCS_NAME=y231-nur40t    ##  <-----------------------------------input
RUN_FILE_NAME=Y-2-3-1-N-UR40T-twcc.sh    ##  <--------------------input
GPU_NUM=2   ##  <--------------------------------------------------input
GIT_PULL=True ##-------------------------------------------------git
APIKEY=f05a9739-fd95-4478-b7a5-f42a0d0b8257
PROJECT_ID=MST111038
RUN_FILE_PATH=call_scripts/train

twccli config init -pcode $PROJECT_ID --apikey $APIKEY

twccli mk ccs -n $CCS_NAME -gpu ${GPU_NUM}m -itype Custom\ Image -img pytorch-21.08-py3:CTCPLM1 -wait -json > $RUN_FILE_NAME.ccs_res.log
twccli ls ccs
CCS_ID=$(cat $RUN_FILE_NAME.ccs_res.log | jq '.id')
echo "==============CCS_ID:$CCS_ID========================="

# CCS_ID=3152233
# IP=203.145.216.196
# PORT=52532
# ssh-copy-id valex1377@$IP -p $PORT

ROOT_PATH=/work/valex1377/CTC_PLM/Reorder_nat
echo "==============Nvidia-smi Info.========================="
ssh -t -o "StrictHostKeyChecking=no" `twccli ls ccs -gssh -s $CCS_ID` "/bin/bash --login -c nvidia-smi"
wait
echo "==============Conda deactivate==================="
ssh -t -o "StrictHostKeyChecking=no" `twccli ls ccs -gssh -s $CCS_ID` "conda deactivate"
wait
echo "==============Conda Env.==================="
ssh -t -o "StrictHostKeyChecking=no" `twccli ls ccs -gssh -s $CCS_ID` "echo $CONDA_DEFAULT_ENV"
wait
if [ "$GIT_PULL" = "True" ]
then
    echo "==============GIT PULL============================"
    ssh -t -o "StrictHostKeyChecking=no" `twccli ls ccs -gssh -s $CCS_ID` "cd $ROOT_PATH; git pull"
fi
wait
echo "==============Run job============================"
ssh -t -o "StrictHostKeyChecking=no" `twccli ls ccs -gssh -s $CCS_ID` "cd $ROOT_PATH; bash $RUN_FILE_PATH/$RUN_FILE_NAME"
wait


cp /home/valex/.twcc_data/credential_$PROJECT_ID /home/valex/.twcc_data/credential
echo "==============Remove CCS========================="
twccli rm ccs -f -s $CCS_ID
rm $RUN_FILE_NAME.ccs_res.log
wait

echo "==============Checking CCS 檢視容器狀態==========="
twccli ls ccs