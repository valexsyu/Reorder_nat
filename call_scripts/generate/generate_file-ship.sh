source $HOME/.bashrc 
conda activate base

# #--------iwslt14 deen main table------
# # --avg-ck-turnoff 
# bash call_scripts/generate_nat.sh -b 1 --data-subset test-valid --no-atten-mask --avg-speed 1 \
#     -e K-2-1-1-H12-UR40M -e K-2-1-1-N-UR40M -e K-2-1-1-N-UR40T -e K-2-1-1-N-UF30T -e K-6-1-1-N-UF30T \
#     -e I-6-1-1-N-UF30T -e 2-2-1-1-H12-UR40M -e 2-2-1-1-N-UR40M -e 2-2-1-1-N-UR40T -e 2-2-1-1-N-UF30T \
#     -e 2-6-1-1-N-UF30T -e J-6-1-1-N-UF30T -e K-2-1-1-H12-UD25M -e 2-2-1-1-H12-UD25M \


#--------wmt14 deen main table------
# --avg-ck-turnoff
#=========New======== 
# hrun -s -N s02 -G -c 20 -m 40 bash call_scripts/generate_nat.sh -b 1 --data-subset test-valid --avg-speed 1 \
# -e Z-2-1-1-N-UF30T

#========Old======== --avg-ck-turnoff
# hrun -s -N s03 -c 20 -m 40 bash call_scripts/generate_nat.sh -b 1 --data-subset test --ck-types top --avg-speed 1 --no-atten-mask \
# -e 2-2-1-1-N-UF20T \



# hrun -c 20 -m 40  bash call_scripts/generate_nat.sh -b 1 --data-subset test --ck-types top-lastk --avg-speed 1 --no-atten-mask \
# -e L-1-1-1-H12-UR40M

# bash call_scripts/generate_nat.sh -b 50 --data-subset test --avg-speed 1 --ck-types top  \
#                                         --avg-ck-turnoff \
#                                         -e m-8-1-1-K12-UF20M-AutoModelForMaskedLM \
#                                         -e m-8-3-1-K12-UF20M-AutoModelForMaskedLM

#  --avg-ck-turnoff \
# bash call_scripts/generate_nat.sh -b 100 --data-subset test --avg-speed 1 --ck-types top  \
#                                         -e m-8-1-1-K12-UF20M-AutoModelForMaskedLM \
#                                         -e m-8-1-1-K12-UF20M-AutoModelForMaskedLM-randPos \
#                                         -e m-8-1-1-K12-UR40M-AutoModelForMaskedLM-randPos \
#                                         -e m-8-1-3-K12-UF20B \
#                                         -e m-8-1-4-K12-UF20B \
#                                         -e m-8-1-4-K12-UF20B \
#                                         -e m-8-1-4-K12-UR40B \
#                                         -e m-8-2-1-K12-UF20M-AutoModelForMaskedLM-randPos \
#                                         -e m-8-2-1-K12-UR40M-AutoModelForMaskedLM-randPos \
#                                         -e m-8-2-3-K12-UF20B \
#                                         -e m-8-2-4-K12-UF20B \
#                                         -e m-8-2-4-K12-UR40B \
#                                         -e m-8-3-1-K12-UF20M-AutoModelForMaskedLM \
#                                         -e m-8-3-1-K12-UF20M-AutoModelForMaskedLM-randPos \
#                                         -e m-8-3-1-K12-UR40M-AutoModelForMaskedLM-randPos \
#                                         -e m-8-3-3-K12-UF20B \
#                                         -e m-8-3-4-K12-UF20B \
#                                         -e m-8-3-4-K12-UR40B \
#                                         -e m-8-4-1-K12-UF20M-AutoModelForMaskedLM-randPos \
#                                         -e m-8-4-1-K12-UR40M-AutoModelForMaskedLM-randPos \
#                                         -e m-8-4-3-K12-UF20B \
#                                         -e m-B-1-1-K12-UF20M-AutoModelForMaskedLM-randPos 

# bash call_scripts/generate_nat.sh -b 100 --data-subset test --avg-speed 1 --ck-types top  \
#                                         -e m-8-2-5-K12-UF20M \
#                                         -e m-8-2-5-K12-UF20B \
#                                         -e m-8-1-4-K12-UF20M \
#                                         -e m-8-2-4-K12-UF20M \
#                                         -e m-8-3-4-K12-UF20M \
#                                         -e m-8-2-1-H12-UF20M \
#                                         -e m-8-4-1-H12-UF20M \
#                                         -e m-8-3-1-H12-UF20M-dp001 \
#                                         -e m-8-3-1-K12-UF20M-dp001 \
#                                         -e m-8-3-3-K12-UF20M-dp001 \
#                                         -e m-8-1-5-K12-UR40B \
#                                         -e m-8-2-5-K12-UR40B \
#                                         -e m-8-4-5-K12-UF20B \
#                                         -e m-8-4-5-K12-UF20M



# bash call_scripts/generate_nat.sh -b 100 --data-subset test  --avg-ck-turnoff \
#                        --ck-types top --avg-speed 1 \
#                         -e m-8-1-3-K12-UF20M-test --visualization                                        
                                    

#1-4 1-5 2-6-7-8 3-4 6-6-7-8
#1-6 #2-9
#1-7 #2-10-11-12 #3-5 6-8-9
#1-1 #3-1 # 6-1-2-3
# bash call_scripts/generate_nat.sh --data-subset test --ck-types top \
#                         -b 100 \
#                         -e m-B-1-1-A12-UF20M-lm5x015 \ˊ
#                         -e m-B-1-3-B12-UF20B-lmx015 \
#                         -e m-B-1-3-C12-UF20M-lm5 \
#                         -e m-B-1-3-B12-UF20B \
#                         -e m-B-1-3-B12-UF20M-lmx015 \
#                         -e m-B-3-3-B12-UF20M-lmx015 \
#                         -e m-B-3-3-B12-UF20B-lmx015 



# bash call_scripts/generate_nat.sh --data-subset test --ck-types top \
#                         -b 100 \
#                         -e K-2-3-3-B12-UD25B-lmx015 \
#                         -e Y-2-3-3-B12-UD25B-lmx015 \
#                         -e 2-2-1-3-B12-UD25B-lmx015 \
#                         -e 2-2-2-3-B12-UD25B-lmx015 \
#                         -e 2-2-3-3-B12-UR40B-lmx015


# # --avg-ck-turnoff
# bash call_scripts/generate_nat.sh --data-subset test --ck-types top \
#                         -b 50 \
#                         -e m-B-3-3-N-UF60M-Ltest \

# bash call_scripts/generate_nat.sh --data-subset test --ck-types top \
#                         --avg-ck-turnoff \
#                         -b 50 \
#                         --debug \
#                         -e m-B-3-1-N-UF30M-Ltest 
#                         # --debug \
#                         # --arch ctcpmlm_rate_pred 
# #                         # --avg-ck-turnoff \
# #                         # --debug \                        

#                         # -e m-B-3-1-N-UF20M-Ltest \
#                         # -e m-B-3-1-N-UF30M-Ltest \

# bash call_scripts/generate_nat.sh --data-subset test --ck-types top \
#                         -b 10 \
#                         --arch ctcpmlm_rate_predictor \
#                         --task transaltion_ctcpmlm_rate \
#                         --criterion nat_ctc_pred_rate_loss \
#                         -e m-B-1-1-N-UR20M-rate_predict        


# bash call_scripts/generate_nat.sh --data-subset test --ck-types top \
#                         -b 50 \
#                         --arch ctcpmlm_rate_selection \
#                         --task translation_ctcpmlm \
#                         --criterion nat_ctc_sel_rate_loss \
#                         -e m-B-1-1-N-UR20M-rate_select


# bash call_scripts/generate_nat.sh --data-subset test --ck-types top \
#                         -b 50 \
#                         --arch ctcpmlm_rate_predictor \
#                         --task transaltion_ctcpmlm_rate \
#                         --criterion nat_ctc_pred_rate_loss \
#                         -e m-B-1-1-N-UR20M-rate_predict_divTGT



bash call_scripts/generate_nat.sh --data-subset test --ck-types top \
                        -b 50 \
                        --task translation_ctcpmlm \
                        --arch nat_pretrained_model \
                        --criterion nat_ctc_loss \
                        -e m-B-1-1-N-UR20M \
                        -e m-B-1-1-N-UR30M-NEW \
                        -e m-B-1-1-N-UR40M-NEW 