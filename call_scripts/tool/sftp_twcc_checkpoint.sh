#!/bin/sh

# local_file=2-2-3-1-N-UF20M
# destination_file=2-2-3-1-N-UF20M


local_file=2-2-3-1-N-UF20M
destination_file=2-2-3-1-N-UF20M

destination_root=/work/valex1377/CTC_PLM/Reorder_nat/checkpoints
local_root=checkpoints/
action=get # get : get data from twcc ; put:put data to twcc

if [ "$action" = "get" ] 
then
    echo "Get Action : Get from Twcc $destination_file to Local $local_root"
    hrun -s -c 4 -m 20 echo "get -r $destination_file ./" | sftp valex1377@xdata1.twcc.ai:$destination_root
elif [ "$action" = "put" ] 
then
    echo "Put Action : Put from Local $local_file to $destination_file"
    hrun -s -c 4 -m 20 echo "put -r $local_path/$local_file $destination_root " | sftp valex1377@xdata1.twcc.ai:$destination_root/$destination_file
else
    echo "Error Action=$action , only [put/get] command"
fi



    