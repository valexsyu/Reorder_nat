# source $HOME/.bashrc 
# conda activate base

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
#                         --debug \
#                         --avg-ck-turnoff \
#                         -e m-B-1-1-N-UR20M-rate_avg


# bash call_scripts/generate_nat.sh --data-subset test --ck-types top \
#                         -b 50 \
                        # --arch ctcpmlm_rate_predictor \
                        # --task transaltion_ctcpmlm_rate \
                        # --criterion nat_ctc_pred_rate_loss \
#                         -e m-B-1-1-N-UR20M-rate_predict_divTGT



# bash call_scripts/generate_nat.sh --data-subset test --ck-types top \
#                         -b 10 \
#                         --task translation_ctcpmlm \
#                         --arch nat_pretrained_model \
#                         --criterion nat_ctc_loss \
#                         --avg-ck-turnoff \
#                         -e 2-2-1-1-H12-UF20M \
#                         -e 2-2-1-1-H12-UR40M \
#                         -e 2-2-1-1-H4-UF20M \
#                         -e 2-2-1-1-H7-UF20M \
#                         -e 2-2-1-1-H7-UF20T \
#                         -e 2-2-1-1-H7-UR40M \
#                         -e 2-2-1-1-N-UF20M \
#                         -e 2-2-1-1-N-UF20T \
#                         -e 2-2-1-1-N-UR40M \
#                         -e 2-2-3-1-H12-UF20M \
#                         -e 2-2-3-1-H4-UF20M \
#                         -e 2-2-3-1-H7-UF20M \
#                         -e 2-2-3-1-N-UF20M \
#                         -e 2-6-1-1-H12-UF20M \
#                         -e 2-6-1-1-H12-UF20T \
#                         -e 2-6-1-1-H7-UF20M \
#                         -e 2-6-1-1-H7-UF20T \
#                         -e 2-6-1-1-N-UF20M \
#                         -e 2-6-1-1-N-UF20T \
#                         -e J-2-1-1-H12-UF20M \
#                         -e J-2-1-1-H12-UF20T \
#                         -e J-2-1-1-H12-UR40M \
#                         -e J-2-1-1-H7-UF20M \
#                         -e J-2-1-1-H7-UF20T \
#                         -e J-2-1-1-H7-UR40M \
#                         -e J-2-1-1-N-UF20M \
#                         -e J-2-1-1-N-UF20T \
#                         -e J-2-1-1-N-UR40M \
#                         -e J-6-1-1-H12-UF20M \
#                         -e J-6-1-1-H7-UF20M \
#                         -e J-6-1-1-H7-UF20T \
#                         -e J-6-1-1-N-UF20M \
#                         -e K-2-1-1-H12-UR40M \
#                         -e K-2-1-1-H7-UR40M \
#                         -e K-2-1-1-N-UR40M \
#                         -e m-B-1-1-H12-UR20M \
#                         -e m-B-1-1-N-UF20M-NEW \
#                         -e m-B-1-1-N-UR20M \
#                         -e m-B-1-1-N-UR30M-NEW \
#                         -e m-B-1-1-N-UR40M-NEW \
#                         -e m-B-3-1-H12-UF20M \
#                         -e m-B-3-1-H12-UF30M \
#                         -e m-B-3-1-H12-UF40M \
#                         -e m-B-3-1-N-UF20M \
#                         -e m-B-3-1-N-UF40M \

                        # --avg-ck-turnoff \

# bash call_scripts/generate_nat.sh --data-subset test --ck-types top \
#                         -b 10 \
#                         --task translation_ctcpmlm \
#                         --arch nat_pretrained_model \
#                         --criterion nat_ctc_loss \
#                         --avg-ck-turnoff \
#                         --debug \
#                         -e 2-2-3-1-H12-UF40T-50k-fixpos

bash call_scripts/generate_nat.sh --data-subset test --ck-types top \
                        -b 10 \
                        --task translation_ctcpmlm \
                        --arch nat_pretrained_model \
                        --criterion nat_ctc_loss \
                        -e 2-2-3-1-H12-UR40M-50k


# bash call_scripts/generate_nat.sh --data-subset test --ck-types last-best-top-lastk \
# batch_size=10
# bash call_scripts/generate_nat.sh --data-subset test --ck-types top \
#                         -b $batch_size \
#                         --arch ctcpmlm_rate_selection \
#                         --task translation_ctcpmlm \
#                         --criterion nat_ctc_avg_rate_loss \
#                         --skip-load-step-num \
#                         -e Z-2-3-1-N-UR30M-rate_avg-33k


# bash call_scripts/generate_nat.sh --data-subset test --ck-types last-best-top-lastk \
# bash call_scripts/generate_nat.sh --data-subset test --ck-types top \
#                         -b 10 \
#                         --skip-exist-genfile \
#                         --skip-load-step-num \
#                         --avg-ck-turnoff \
#                         --task translation_ctcpmlm \
#                         --arch nat_pretrained_model \
#                         --criterion nat_ctc_loss \
#                         -e m-B-3-1-N-UR20M \
#                         -e m-B-3-1-N-UR25M \
#                         -e m-B-3-1-N-UR30M \
#                         -e m-B-3-1-N-UR35M \
#                         -e m-B-3-1-N-UR40M \
#                         -e v-I-3-1-N-UR20M \
#                         -e v-I-3-1-N-UR30M \
#                         -e v-I-3-1-N-UR40M \
#                         -e 2-2-3-1-N-UR20M \
#                         -e 2-2-3-1-N-UR30M \
#                         -e 2-2-3-1-N-UR40M \
#                         -e K-2-3-1-N-UR20M \
#                         -e K-2-3-1-N-UR30M \
#                         -e K-2-3-1-N-UR40M \
#                         -e r-E-3-1-N-UR20M \
#                         -e r-E-3-1-N-UR30M \
#                         -e r-E-3-1-N-UR40M \
#                         -e s-F-3-1-N-UR20M \
#                         -e s-F-3-1-N-UR30M \
#                         -e s-F-3-1-N-UR40M \
#                         -e r-E-3-1-N-UR20M-100k_300k \
#                         -e r-E-3-1-N-UR30M-100k_300k \
#                         -e r-E-3-1-N-UR40M-100k_300k \
#                         -e s-F-3-1-N-UR20M-100k_300k \
#                         -e s-F-3-1-N-UR30M-100k_300k \
#                         -e s-F-3-1-N-UR40M-100k_300k \
#                         -e t-G-3-1-N-UR20M \
#                         -e t-G-3-1-N-UR30M \
#                         -e t-G-3-1-N-UR40M \
#                         -e u-H-3-1-N-UR20M \
#                         -e u-H-3-1-N-UR30M \
#                         -e u-H-3-1-N-UR40M 



# # # "2-2-3-1-N-UR30M-rate_avg-33k" "K-2-3-1-N-UR20M-rate_avg-33k" "K-2-3-1-N-UR30M-rate_avg-33k" "K-2-3-1-N-UR30M-rate_avg-100k"
# experiments=("Z-2-3-1-N-UR30M-rate_avg-33k")
# rate_list=(2.0 3.0 4.0)
# # rate_list=(2.5 3.5)
# for experiment_id in "${experiments[@]}"; do
#     CHECKPOINT=checkpoints/$experiment_id
#     TOPK=5
#     avg_topk_best_checkpoints $CHECKPOINT $TOPK $CHECKPOINT/checkpoint_best_top$TOPK.pt
#     for debug_value in "${rate_list[@]}"; do
#         echo "=============================================="
#         echo "=====  $experiment_id with Rate: $debug  ========="
#         echo "=============================================="
#         batch_size=10
#         echo "debug_value = $debug_value"
#         CUDA_VISIBLE_DEVICES=0 bash call_scripts/generate_nat.sh --data-subset test \
#                             --ck-types top --avg-speed 1 \
#                                 -b $batch_size \
#                                 --arch ctcpmlm_rate_selection \
#                                 --task translation_ctcpmlm \
#                                 --criterion nat_ctc_avg_rate_loss \
#                                 --debug \
#                                 --skip-load-step-num \
#                                 --avg-ck-turnoff \
#                                 --debug-value $debug_value \
#                                 -e $experiment_id
#         mv checkpoints/$experiment_id/test/best_top5_${batch_size}_1.bleu/generate-test.txt \
#         checkpoints/$experiment_id/test/best_top5_${batch_size}_1.bleu/generate-test-$debug_value.txt 
#     done  
# done




