source $HOME/.bashrc 
conda activate base

# bash call_scripts/train_nat.sh -e 2-2-1-1-H7-UR40M --fp16 -g 4 --save-interval-updates 70000 --max-tokens 1536
# bash call_scripts/generate_nat.sh -e 2-2-1-1-H7-UR40M
bash call_scripts/train_nat.sh -e E-2-1-1-H12-UF30M --fp16 -g 4 --save-interval-updates 100000 --max-update 300000 --lm-start-step 200000 --max-tokens 2048 -b 65536
bash call_scripts/generate_nat.sh -e E-2-1-1-H12-UF30M --data-subset test-valid ## atten_mask added
bash call_scripts/train_nat.sh -e E-2-1-1-H12-UR40M --fp16 -g 4 --save-interval-updates 100000 --max-update 300000 --lm-start-step 200000 --max-tokens 2048 -b 65536
bash call_scripts/generate_nat.sh -e E-2-1-1-H12-UR40M --data-subset test-valid ## atten_mask added
bash
# bash call_scripts/train_nat.sh -e 2-2-1-1-H12-UD45T --fp16 -g 4 --save-interval-updates 70000 --max-tokens 1536
# bash call_scripts/generate_nat.sh -e 2-2-1-1-H12-UD45T 
# bash call_scripts/train_nat.sh -e 2-2-1-1-H12-UD40T --fp16 -g 4 --save-interval-updates 70000 --max-tokens 1536
# bash call_scripts/generate_nat.sh -e 2-2-1-1-H12-UD40T 
# bash call_scripts/train_nat.sh -e 2-2-1-1-H12-UF50T --fp16 -g 4 --save-interval-updates 70000 --max-tokens 1536
# bash call_scripts/generate_nat.sh -e 2-2-1-1-H12-UF50T 
# bash call_scripts/train_nat.sh -e 2-2-1-1-H12-UF40T --fp16 -g 4 --save-interval-updates 70000 --max-tokens 1536
# bash call_scripts/generate_nat.sh -e 2-2-1-1-H12-UF40T 

