bash call_scripts/train_nat.sh -e K-2-1-1-H12-UF30M --fp16 --save-interval-updates 70000 --max-tokens 2048 -g 2
bash call_scripts/generate_nat.sh -e K-2-1-1-H12-UF30M --data-subset test-valid 