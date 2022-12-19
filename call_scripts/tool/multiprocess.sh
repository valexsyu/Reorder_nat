# task(){
#    sleep 1; echo "$1";
#    # echo "QQ"
# }
# # for thing in a b c d e f g; do 
# #    task "$thing"
# # done

# N=5
# num_lines=25
# # (
# # for (( c=1; c<=$num_lines; c++ )); do 
# #    ((i=i%N)); ((i++==0)) && wait
# #    task "$c" & 
# # done
# # )


# for (( c=1; c<=$num_lines; c++ ))
# do
#     # Every THREADSth job, stop and wait for everything
#     # to complete.
#     if (( i %N == 0 )); then
#         wait
#     fi
#     ((i++))
#     echo "QQ" &
#     task $c &
#     wait
# done

# task(){
#    sleep 1; echo "$1";
#    # echo "QQ"
# }



#!/bin/bash 

sleep 10 & 
echo ${!} 
sleep 600 & 
echo ${!} 
sleep 1200 & 
echo ${!} 
sleep 1800 & 
echo ${!} 
sleep 3600 & 
echo ${!}