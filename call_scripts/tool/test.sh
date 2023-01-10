# function get_twcc_file() {
# experiment=$1
# twcc_root=/work/valex1377/CTC_PLM/Reorder_nat
# sftp valex1377@xdata1.twcc.ai << EOF
#   get $twcc_root/checkpoints/ checkpoints/$experiment/
#   get $twcc_root/checkpoints/ checkpoints/$experiment/
#   get $twcc_root/checkpoints/ checkpoints/$experiment/
# EOF  

# }
# experiment=test_folder

# get_twcc_file $experiment

destination_root=/work/valex1377/CTC_PLM/Reorder_nat/checkpoints
destination_file=test_folder
echo "get -r $destination_file ./ " | hrun -s -c 4 -m 20 sftp valex1377@xdata1.twcc.ai:$destination_root

 