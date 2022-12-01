destination_path=/work/valex1377/CTC_PLM/nat_data/
local_path=checkpoints/test_folder
hrun -s -c 4 -m 20 echo "put -r $local_path" | sftp valex1377@xdata1.twcc.ai:$destination_path