CUDA_VISIABLE_DEVICES=0 bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-H12-UR25M --twcc --fp16 --save-interval-updates 70000 --max-tokens 3072
CUDA_VISIABLE_DEVICES=0 bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-H12-UR22M --twcc --fp16 --save-interval-updates 70000 --max-tokens 3072
CUDA_VISIABLE_DEVICES=0 bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-H12-UR20M --twcc --fp16 --save-interval-updates 70000 --max-tokens 4096
# CUDA_VISIABLE_DEVICES=0 bash call_scripts/iwslt14_de-en/training/train_nat.sh -e 2-2-1-1-H12-UR15M --twcc --fp16 --save-interval-updates 70000 --max-tokens 4096

CUDA_VISIABLE_DEVICES=0 bash call_scripts/iwslt14_de-en/inference/generate_nat.sh -e 2-2-1-1-N-UD45T -e 2-2-1-1-N-UD40T -e 2-2-1-1-N-UD45M -e 2-2-1-1-H12-UD45M -e 2-2-1-1-H12-UR22M -e 2-2-1-1-H12-UR20M
echo "========================end========================="