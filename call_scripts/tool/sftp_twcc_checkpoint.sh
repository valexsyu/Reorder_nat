#!/bin/sh

local_file=$1
destination_file=$1

destination_root=/work/valex1377/CTC_PLM/Reorder_nat/checkpoints
local_root=checkpoints
action=$2 # get : get data from twcc ; put:put data to twcc

if [ "$action" = "get" ] 
then
    echo "Get Action : Get from Twcc $destination_file to Local $local_root"
    echo "get -r $destination_file $local_root/" | hrun -s -c 4 -m 20 sftp valex1377@xdata1.twcc.ai:$destination_root
elif [ "$action" = "put" ] 
then
    echo "Put Action : Put from Local $local_file to $destination_file"
    echo "put -r $local_path/$local_file ./ " | hrun -s -c 4 -m 20 sftp valex1377@xdata1.twcc.ai:$destination_root/
else
    echo "Error Action=$action , only [put/get] command"
fi



    