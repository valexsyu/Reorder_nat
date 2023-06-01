# experiment_1=1-1-4-1-N-UF20T-new
# CUDA_VISIBLE_DEVICES=0 bash call_scripts/train_nat.sh -e $experiment_1 --local --fp16 --save-interval-updates 70000 --max-update 100000 --max-tokens 1024 -b 12288 -g 1 --dropout 0.1 --no-atten-mask
# CUDA_VISIBLE_DEVICES=0 bash call_scripts/generate_nat.sh -b 1 --local --data-subset test --ck-types top --avg-speed 1 --no-atten-mask --avg-ck-turnoff --skip-exist-genfile --load-exist-bleu \
#     -e $experiment_1 \

# experiment_1=m-8-2-1-K12-UF20M-AutoModelForMaskedLM
# CUDA_VISIBLE_DEVICES=0 bash call_scripts/train_nat.sh -e $experiment_1 \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --fp16
# CUDA_VISIBLE_DEVICES=0 bash call_scripts/generate_nat.sh -b 1 --local --data-subset test --ck-types top --avg-speed 1 --no-atten-mask --avg-ck-turnoff --skip-exist-genfile --load-exist-bleu \
#     -e $experiment_1 \

# experiment_1=m-8-1-4-K12-UF20B-drop001-test
# CUDA_VISIBLE_DEVICES=0 bash call_scripts/train_nat.sh -e $experiment_1 --local \
#                                --save-interval-updates 70000 --max-tokens 1024 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1 --dryrun --dropout 0.01

# bash call_scripts/train_nat.sh -e m-8-2-5-K12-UF20M \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 2 --fp16                               

# experiment_1=m-8-1-1-A12-UF20M
# CUDA_VISIBLE_DEVICES=0 bash call_scripts/train_nat.sh -e $experiment_1 --local \
#                                --save-interval-updates 70000 --max-tokens 2048 \
#                                --has-eos --max-update 100000 --lm-start-step 75000 \
#                                --g 1

# CUDA_VISIBLE_DEVICES=0 bash call_scripts/train_nat.sh -e m-B-1-3-N-UR20M-rate-pred --local \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --has-eos --max-update 100000 \
#                                 --arch ctcpmlm_rate_selection \
#                                 --fp16 --g 1  --debug --dryrun --valid-set     



# bash call_scripts/train_nat.sh -e m-B-1-1-N-UR20M \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --task translation_ctcpmlm \
#                                 --arch nat_pretrained_model \
#                                 --criterion nat_ctc_loss \
#                                 --has-eos --max-update 100000 \
#                                 --hydra \
#                                 --local \
#                                 -g 1 --fp16   




# bash call_scripts/train_nat.sh -e m-B-1-1-N-UR30M-hydra \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --task translation_ctcpmlm \
#                                 --arch nat_pretrained_model \
#                                 --criterion nat_ctc_loss \
#                                 --has-eos --max-update 100000 \
#                                 --hydra \
#                                 --local \
#                                 -g 1 --fp16  




# bash call_scripts/train_nat.sh -e m-B-1-1-N-UR20M \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --task translation_ctcpmlm \
#                                 --arch nat_pretrained_model \
#                                 --criterion nat_ctc_loss \
#                                 --has-eos --max-update 100000 \
#                                 --hydra \
#                                 --local \
#                                 -g 1 --fp16   




# bash call_scripts/train_nat.sh -e m-B-1-1-N-UR20M \
#                                 --save-interval-updates 70000 --max-tokens 2048 \
#                                 --task translation_ctcpmlm \
#                                 --arch nat_pretrained_model \
#                                 --criterion nat_ctc_loss \
#                                 --has-eos --max-update 100000 \
#                                 --hydra \
#                                 --local \
#                                 -g 1 --fp16   



# source call_scripts/train/pair_experiment.sh
# pair_experiment_iwslt14_3080x1_768_50k_loacl J-2-3-1-N-UR40T J-2-3-1-H12-UR40T

