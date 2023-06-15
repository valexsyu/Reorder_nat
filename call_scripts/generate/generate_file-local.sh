source $HOME/.bashrc 
conda activate base

# #--skip-exist-genfile --load-exist-bleu 
# CUDA_VISIBLE_DEVICES=1 bash call_scripts/generate_nat.sh -b 50 --local --data-subset valid --ck-types top --avg-speed 1 --no-atten-mask --avg-ck-turnoff --load-exist-bleu \
# -e 2-2-1-1-N-UF20T \
# -e 2-2-1-1-H12-UF20T \
# -e 2-2-1-1-H11-UF20T \
# -e 2-2-1-1-H10-UF20T \
# -e 2-2-1-1-H9-UF20T \
# -e 2-2-1-1-H8-UF20T \
# -e 2-2-1-1-H7-UF20T \
# -e 2-2-1-1-H6-UF20T \
# -e 2-2-1-1-H5-UF20T \
# -e 2-2-1-1-H4-UF20T \
# -e 2-2-1-1-H3-UF20T \
# -e 2-2-1-1-H2-UF20T \
# -e 2-2-1-1-H1-UF20T \
# -e 2-2-1-1-N-UF20M   \
# -e 2-2-1-1-H12-UF20M \
# -e 2-2-1-1-H11-UF20M \
# -e 2-2-1-1-H10-UF20M \
# -e 2-2-1-1-H9-UF20M  \
# -e 2-2-1-1-H8-UF20M  \
# -e 2-2-1-1-H7-UF20M  \
# -e 2-2-1-1-H6-UF20M  \
# -e 2-2-1-1-H5-UF20M  \
# -e 2-2-1-1-H4-UF20M  \
# -e 2-2-1-1-H3-UF20M  \
# -e 2-2-1-1-H2-UF20M  \
# -e 2-2-1-1-H1-UF20M  
# -e 1-1-1-1-N-UF20T   \
# -e 1-1-1-1-H12-UF20T \
# -e 1-1-1-1-H11-UF20T \
# -e 1-1-1-1-H10-UF20T \
# -e 1-1-1-1-H9-UF20T  \
# -e 1-1-1-1-H8-UF20T  \
# -e 1-1-1-1-H7-UF20T  \
# -e 1-1-1-1-H6-UF20T  \
# -e 1-1-1-1-H5-UF20T  \
# -e 1-1-1-1-H4-UF20T  \
# -e 1-1-1-1-H3-UF20T  \
# -e 1-1-1-1-H2-UF20T  \
# -e 1-1-1-1-H1-UF20T  \
# -e 1-1-1-1-N-UF20M  \
# -e 1-1-1-1-H12-UF20M \
# -e 1-1-1-1-H11-UF20M \
# -e 1-1-1-1-H10-UF20M \
# -e 1-1-1-1-H9-UF20M  \
# -e 1-1-1-1-H8-UF20M  \
# -e 1-1-1-1-H7-UF20M  \
# -e 1-1-1-1-H6-UF20M  \
# -e 1-1-1-1-H5-UF20M  \
# -e 1-1-1-1-H4-UF20M  \
# -e 1-1-1-1-H3-UF20M  \
# -e 1-1-1-1-H2-UF20M  \
# -e 1-1-1-1-H1-UF20M  
# mv call_scripts/generate/output_file/output_read.csv call_scripts/generate/output_file/output_read_layer.csv 


# # --skip-exist-genfile --load-exist-bleu 
# CUDA_VISIBLE_DEVICES=1 bash call_scripts/generate_nat.sh -b 1 --local --data-subset test --ck-types top --avg-speed 1 --no-atten-mask --avg-ck-turnoff --load-exist-bleu \
# -e 1-1-1-1-H12-UF20T \
# -e 2-2-1-1-H12-UF20T \
# -e 5-3-1-1-H12-UF20T \
# -e 7-4-1-1-H12-UF20T \
# -e 1-5-1-1-H12-UF20T \
# -e 3-1-1-1-H12-UF20T \
# -e 4-2-1-1-H12-UF20T \
# -e 6-3-1-1-H12-UF20T \
# -e 8-4-1-1-H12-UF20T \
# -e 3-5-1-1-H12-UF20T 
# mv call_scripts/generate/output_file/output_read.csv call_scripts/generate/output_file/output_read_model.csv 


# CUDA_VISIBLE_DEVICES=1 bash call_scripts/generate_nat.sh -b 1 --local --data-subset test --ck-types top --avg-speed 1 --no-atten-mask --avg-ck-turnoff --load-exist-bleu \
# -e J-6-4-1-N-UF20T \
# -e J-6-4-1-N-UF20M \
# -e 2-6-4-1-N-UF20T \
# -e 2-6-4-1-N-UF20M \
# -e J-2-1-1-N-UF20T \
# -e J-2-1-1-N-UF20M \
# -e 2-2-1-1-N-UF20T \
# -e 2-2-1-1-N-UF20M \
# -e 2-6-4-1-H12-UF20T \
# -e 2-6-4-1-H12-UF20M \
# -e J-6-4-1-H12-UF20T \
# -e J-6-4-1-H12-UF20M \
# -e 2-2-1-1-H12-UF20T \
# -e 2-2-1-1-H12-UF20M \
# -e J-2-1-1-H12-UF20T \
# -e J-2-1-1-H12-UF20M 
# mv call_scripts/generate/output_file/output_read.csv call_scripts/generate/output_file/output_read_ablation.csv 

# CUDA_VISIBLE_DEVICES=1 bash call_scripts/generate_nat.sh -b 50 --local --data-subset valid --ck-types top --avg-speed 1 --no-atten-mask --avg-ck-turnoff --load-exist-bleu \
# -e 2-2-1-1-H12-UF20T \
# -e 2-2-2-1-H12-UF20T \
# -e 2-2-3-1-H12-UF20T \
# -e 2-2-4-1-H12-UF20T \
# -e 2-2-1-1-H12-UF20M \
# -e 2-2-2-1-H12-UF20M \
# -e 2-2-3-1-H12-UF20M \
# -e 2-2-4-1-H12-UF20M \
# -e 2-2-1-1-N-UF20T \
# -e 2-2-2-1-N-UF20T \
# -e 2-2-3-1-N-UF20T \
# -e 2-2-4-1-N-UF20T \
# -e 2-2-1-1-N-UF20M \
# -e 2-2-2-1-N-UF20M \
# -e 2-2-3-1-N-UF20M \
# -e 2-2-4-1-N-UF20M 
# mv call_scripts/generate/output_file/output_read.csv call_scripts/generate/output_file/output_read_freeze.csv 

# CUDA_VISIBLE_DEVICES=1 bash call_scripts/generate_nat.sh -b 50 --local --data-subset valid --ck-types top --avg-speed 1 --no-atten-mask --avg-ck-turnoff --load-exist-bleu \
# -e 2-2-1-1-H12-UR50M \
# -e 2-2-1-1-H12-UR45M \
# -e 2-2-1-1-H12-UR40M \
# -e 2-2-1-1-H12-UR33M \
# -e 2-2-1-1-H12-UR30M \
# -e 2-2-1-1-H12-UR25M \
# -e 2-2-1-1-H12-UR22M \
# -e 2-2-1-1-H12-UR20M \
# -e 2-2-1-1-H12-UR15M \
# -e 2-2-1-1-H12-UD50T \
# -e 2-2-1-1-H12-UD45T \
# -e 2-2-1-1-H12-UD40T \
# -e 2-2-1-1-H12-UD33T \
# -e 2-2-1-1-H12-UD30T \
# -e 2-2-1-1-H12-UD25T \
# -e 2-2-1-1-H12-UD22T \
# -e 2-2-1-1-H12-UD20T \
# -e 2-2-1-1-H12-UD15T \
# -e 2-2-1-1-H12-UD50M \
# -e 2-2-1-1-H12-UD45M \
# -e 2-2-1-1-H12-UD40M \
# -e 2-2-1-1-H12-UD33M \
# -e 2-2-1-1-H12-UD30M \
# -e 2-2-1-1-H12-UD25M \
# -e 2-2-1-1-H12-UD22M \
# -e 2-2-1-1-H12-UD20M \
# -e 2-2-1-1-H12-UD15M \
# -e 2-2-1-1-N-UD50M \
# -e 2-2-1-1-N-UD45M \
# -e 2-2-1-1-N-UD40M \
# -e 2-2-1-1-N-UD33M \
# -e 2-2-1-1-N-UD30M \
# -e 2-2-1-1-N-UD25M \
# -e 2-2-1-1-N-UD22M \
# -e 2-2-1-1-N-UD20M \
# -e 2-2-1-1-N-UD15M \
# -e 2-2-1-1-N-UD50T \
# -e 2-2-1-1-N-UD45T \
# -e 2-2-1-1-N-UD40T \
# -e 2-2-1-1-N-UD33T \
# -e 2-2-1-1-N-UD30T \
# -e 2-2-1-1-N-UD25T \
# -e 2-2-1-1-N-UD22T \
# -e 2-2-1-1-N-UD20T \
# -e 2-2-1-1-N-UD15T 
# mv call_scripts/generate/output_file/output_read.csv call_scripts/generate/output_file/output_read_ratio.csv 


# CUDA_VISIBLE_DEVICES=1 bash call_scripts/generate_nat.sh -b 1 --local --data-subset test --ck-types top --avg-speed 1 --no-atten-mask --avg-ck-turnoff --load-exist-bleu \
# -e J-6-4-1-N-UF20T \
# -e J-6-4-1-N-UF20M \
# -e 2-6-4-1-N-UF20T \
# -e 2-6-4-1-N-UF20M \
# -e J-2-1-1-N-UF20T \
# -e J-2-1-1-N-UF20M \
# -e 2-2-1-1-N-UF20T \
# -e 2-2-1-1-N-UF20M \
# -e 2-6-4-1-H12-UF20T \
# -e 2-6-4-1-H12-UF20M \
# -e J-6-4-1-H12-UF20T \
# -e J-6-4-1-H12-UF20M \
# -e 2-2-1-1-H12-UF20T \
# -e 2-2-1-1-H12-UF20M \
# -e J-2-1-1-H12-UF20T \
# -e J-2-1-1-H12-UF20M 
# mv call_scripts/generate/output_file/output_read.csv call_scripts/generate/output_file/output_read_ablation.csv 

# CUDA_VISIBLE_DEVICES=1 bash call_scripts/generate_nat.sh -b 1 --local --data-subset test --ck-types top --avg-speed 1 --no-atten-mask --avg-ck-turnoff --load-exist-bleu \
# -e 2-2-1-1-N-UF20T \
# -e 2-2-1-1-H12-UF20T \
# -e 2-2-1-1-H11-UF20T \
# -e 2-2-1-1-H10-UF20T \
# -e 2-2-1-1-H9-UF20T \
# -e 2-2-1-1-H8-UF20T \
# -e 2-2-1-1-H7-UF20T \
# -e 2-2-1-1-H6-UF20T \
# -e 2-2-1-1-H5-UF20T \
# -e 2-2-1-1-H4-UF20T \
# -e 2-2-1-1-H3-UF20T \
# -e 2-2-1-1-H2-UF20T \
# -e 2-2-1-1-H1-UF20T \
# -e 2-2-1-1-N-UF20M   \
# -e 2-2-1-1-H12-UF20M \
# -e 2-2-1-1-H11-UF20M \
# -e 2-2-1-1-H10-UF20M \
# -e 2-2-1-1-H9-UF20M  \
# -e 2-2-1-1-H8-UF20M  \
# -e 2-2-1-1-H7-UF20M  \
# -e 2-2-1-1-H6-UF20M  \
# -e 2-2-1-1-H5-UF20M  \
# -e 2-2-1-1-H4-UF20M  \
# -e 2-2-1-1-H3-UF20M  \
# -e 2-2-1-1-H2-UF20M  \
# -e 2-2-1-1-H1-UF20M  \
# -e 1-1-1-1-N-UF20T   \
# -e 1-1-1-1-H12-UF20T \
# -e 1-1-1-1-H11-UF20T \
# -e 1-1-1-1-H10-UF20T \
# -e 1-1-1-1-H9-UF20T  \
# -e 1-1-1-1-H8-UF20T  \
# -e 1-1-1-1-H7-UF20T  \
# -e 1-1-1-1-H6-UF20T  \
# -e 1-1-1-1-H5-UF20T  \
# -e 1-1-1-1-H4-UF20T  \
# -e 1-1-1-1-H3-UF20T  \
# -e 1-1-1-1-H2-UF20T  \
# -e 1-1-1-1-H1-UF20T  \
# -e 1-1-1-1-N-UF20M  \
# -e 1-1-1-1-H12-UF20M \
# -e 1-1-1-1-H11-UF20M \
# -e 1-1-1-1-H10-UF20M \
# -e 1-1-1-1-H9-UF20M  \
# -e 1-1-1-1-H8-UF20M  \
# -e 1-1-1-1-H7-UF20M  \
# -e 1-1-1-1-H6-UF20M  \
# -e 1-1-1-1-H5-UF20M  \
# -e 1-1-1-1-H4-UF20M  \
# -e 1-1-1-1-H3-UF20M  \
# -e 1-1-1-1-H2-UF20M  \
# -e 1-1-1-1-H1-UF20M  
# mv call_scripts/generate/output_file/output_read.csv call_scripts/generate/output_file/output_read__test_layer.csv

# CUDA_VISIBLE_DEVICES=0 bash call_scripts/generate_nat.sh -b 1 --local --data-subset test --ck-types top --avg-speed 1 --no-atten-mask \
# -e a-6-1-1-N-UF30T \
# -e 2-6-4-1-N-UF30T \
# -e J-6-4-1-N-UF30T \
# -e 4-2-1-1-H12-UD25M \
# -e a-2-1-1-H12-UR40M \
# -e b-2-1-1-H12-UR40M \
# -e K-6-4-1-N-UF30T \
# -e I-6-4-1-N-UF30T \

# CUDA_VISIBLE_DEVICES=0 bash call_scripts/generate_nat.sh -b 50 --local --data-subset test --ck-types top --avg-speed 1 --load-exist-bleu \
# -e m-8-1-1-K12-UF20M-AutoModelForMaskedLM \
# -e m-8-3-1-K12-UF20M-AutoModelForMaskedLM \
# -e m-8-2-1-K12-UF20M-AutoModelForMaskedLM \
# -e m-8-4-1-K12-UF20M-AutoModelForMaskedLM \
# -e m-8-1-1-K12-UF20M-AutoModelForMaskedLM-randPos \
# -e m-8-2-1-K12-UF20M-AutoModelForMaskedLM-randPos \
# -e m-8-3-1-K12-UF20M-AutoModelForMaskedLM-randPos \
# -e m-8-4-1-K12-UF20M-AutoModelForMaskedLM-randPos 

# CUDA_VISIBLE_DEVICES=0 bash call_scripts/generate_nat.sh -b 20 --local --data-subset test \
#                        --ck-types top --avg-speed 1 \
#                         -e m-B-1-1-K12-UF20M-AutoModelForMaskedLM-randPos
# -e m-8-1-1-K12-UR40M-AutoModelForMaskedLM-randPos \
# -e m-8-2-1-K12-UR40M-AutoModelForMaskedLM-randPos \
# -e m-8-3-1-K12-UR40M-AutoModelForMaskedLM-randPos \
# -e m-8-4-1-K12-UR40M-AutoModelForMaskedLM-randPos 

# CUDA_VISIBLE_DEVICES=0 bash call_scripts/generate_nat.sh -b 20 --local --data-subset test \
#                        --ck-types top --avg-speed 1 \
#                         -e m-8-3-3-K12-UF20M-test --visualization


# CUDA_VISIBLE_DEVICES=1 bash call_scripts/generate_nat.sh --local --data-subset test \
#                        --ck-types top --avg-speed 1 \
#                         -b 10 \
#                         --task translation_ctcpmlm \
#                         --arch nat_pretrained_model \
#                         --criterion nat_ctc_loss \
#                         --local \
#                         -e Y-2-3-1-N-UR40T    


# bash call_scripts/generate_nat.sh --data-subset test --ck-types top \
#                         -b 10 --local \
#                         --arch ctcpmlm_rate_selection \
#                         --task translation_ctcpmlm \
#                         --criterion nat_ctc_sel_rate_loss \
#                         -e m-B-1-1-N-UR20M-rate_select


# bash call_scripts/generate_nat.sh --data-subset test --ck-types top \
#                         -b 10 --local \
#                         --arch ctcpmlm_rate_predictor \
#                         --task transaltion_ctcpmlm_rate \
#                         --criterion nat_ctc_pred_rate_loss \
#                         --avg-ck-turnoff \
#                         --debug \
#                         -e m-B-1-1-N-UR20M-predsel-rate



# CUDA_VISIBLE_DEVICES=0 bash call_scripts/generate_nat.sh --local --data-subset test \
#                        --ck-types top --avg-speed 1 \
#                         -b 20 \
#                         --arch ctcpmlm_rate_predictor \
#                         --task transaltion_ctcpmlm_rate \
#                         --criterion nat_ctc_pred_rate_loss \
#                         --debug \
#                         -e m-B-1-1-N-UF20M-NEW



# CUDA_VISIBLE_DEVICES=0 bash call_scripts/generate_nat.sh --data-subset test --ck-types top \
#                         -b 10 \
#                         --arch ctcpmlm_rate_selection \
#                         --task transaltion_ctcpmlm_rate \
#                         --criterion nat_ctc_pred_rate_loss \
#                         --local \
#                         --avg-ck-turnoff \
#                         -e 2-2-1-1-H12-UR40M \
#                         -e 2-2-1-1-H7-UF20M 


#=======================================================================================
# function avg_topk_best_checkpoints(){
# 	python scripts/average_checkpoints.py \
# 	            --inputs $1 \
# 				--num-epoch-checkpoints $2 --output $3 \
# 				--ckpref checkpoints.best_bleu	
# }

# # "2-2-3-1-N-UR30M-rate_avg-33k" "K-2-3-1-N-UR20M-rate_avg-33k"
# experiments=("s-F-3-1-N-UR30M-rate_avg-33k-warmup3k")
# rate_list=(2.0 3.0 4.0)
# for experiment_id in "${experiments[@]}"; do
#     CHECKPOINT=checkpoints/$experiment_id
#     TOPK=5
#     avg_topk_best_checkpoints $CHECKPOINT $TOPK $CHECKPOINT/checkpoint_best_top$TOPK.pt
#     for debug_value in "${rate_list[@]}"; do
#         echo "=============================================="
#         echo "=====  $experiment_id with Rate: $debug  ========="
#         echo "=============================================="
#         batch_size=10
#         # experiment_id=m-B-3-1-N-UR30M-rate_avg-33k_warm33
#         echo "debug_value = $debug_value"
#         CUDA_VISIBLE_DEVICES=0 bash call_scripts/generate_nat.sh --local --data-subset test \
#                             --ck-types top --avg-speed 1 \
#                                 -b $batch_size \
#                                 --arch ctcpmlm_rate_selection \
#                                 --task translation_ctcpmlm \
#                                 --criterion nat_ctc_avg_rate_loss \
#                                 --debug \
#                                 --avg-ck-turnoff \
#                                 --skip-load-step-num \
#                                 --debug-value $debug_value \
#                                 -e $experiment_id
#         mv checkpoints/$experiment_id/test/best_top5_${batch_size}_1.bleu/generate-test.txt \
#         checkpoints/$experiment_id/test/best_top5_${batch_size}_1.bleu/generate-test-$debug_value.txt 
#     done  
# done
#=======================================================================================

# --avg-ck-turnoff \
CUDA_VISIBLE_DEVICES=0 bash call_scripts/generate_nat.sh --local --data-subset test \
                       --ck-types top --avg-speed 1 \
                        -b 10 \
                        --task translation_ctcpmlm \
                        --arch nat_pretrained_model \
                        --criterion nat_ctc_loss \
                        --skip-load-step-num \
                        --local \
                        -e t-G-3-1-N-UR20M           





# CUDA_VISIBLE_DEVICES=0 bash call_scripts/generate_nat.sh --local --data-subset test \
#                        --ck-types top --avg-speed 1 \
#                         -b 10 \
#                         --arch ctcpmlm_rate_selection \
#                         --task translation_ctcpmlm \
#                         --criterion nat_ctc_avg_rate_loss \
#                         --avg-ck-turnoff \
#                         --skip-load-step-num \
#                         --local \
#                         -e s-F-3-1-N-UR30M-rate_avg-33k-warmup3k
