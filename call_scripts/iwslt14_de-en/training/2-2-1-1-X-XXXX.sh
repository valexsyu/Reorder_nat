source $HOME/.bashrc 
conda activate base

# bash call_scripts/train_nat.sh -e 2-2-1-1-N-UF50T --fp16 --save-interval-updates 70000
# bash call_scripts/train_nat.sh -e 1-1-1-1-H6-UF20T --fp16 --save-interval-updates 70000 --fp16 --max-tokens 3072
# bash call_scripts/generate_nat.sh -e 2-2-1-1-N-UF50T -e 1-1-1-1-H6-UF20T
bash call_scripts/train_nat.sh -e 1-1-1-1-H9-UF20T --fp16 --save-interval-updates 70000 --max-tokens 4096
bash call_scripts/generate_nat.sh -e 1-1-1-1-H9-UF20T
bash call_scripts/train_nat.sh -e 2-2-1-1-N-UD50T --fp16 --save-interval-updates 70000
bash call_scripts/generate_nat.sh -e 2-2-1-1-N-UD50T
bash call_scripts/train_nat.sh -e 2-2-1-1-H12-UD45T --fp16 --save-interval-updates 70000 --max-tokens 2048
bash call_scripts/generate_nat.sh -e 2-2-1-1-H12-UD45T #atten_mask added
bash call_scripts/train_nat.sh -e 2-2-1-1-H12-UF50T --fp16 --save-interval-updates 70000 --max-tokens 2048
bash call_scripts/generate_nat.sh -e 2-2-1-1-H12-UF50T 
bash call_scripts/train_nat.sh -e 2-2-1-1-H12-UD40T --fp16 --save-interval-updates 70000 --max-tokens 2048
bash call_scripts/generate_nat.sh -e 2-2-1-1-H12-UD40T 
 

