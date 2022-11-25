source $HOME/.bashrc 
conda activate base
bash call_scripts/train_nat.sh -e 2-6-1-1-N-UF30T --fp16 --save-interval-updates 70000 --max-tokens 2048 --no-atten-mask
bash call_scripts/generate_nat.sh -e 2-6-1-1-N-UF30T -b 1 --data-subset test-valid --no-atten-mask
bash call_scripts/train_nat.sh -e J-6-1-1-N-UF30T --fp16 --save-interval-updates 70000 --max-tokens 2048 --no-atten-mask
bash call_scripts/generate_nat.sh -e J-6-1-1-N-UF30T -b 1 --data-subset test-valid --no-atten-mask
bash call_scripts/train_nat.sh -e K-2-1-1-H12-UD25M --fp16 --save-interval-updates 70000 --max-tokens 2048 --no-atten-mask
bash call_scripts/generate_nat.sh -e K-2-1-1-H12-UD25M -b 1 --data-subset test-valid --no-atten-mask