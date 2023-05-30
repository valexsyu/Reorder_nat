source $HOME/.bashrc 
conda activate base

source call_scripts/train/pair_experiment.sh

# pair_experiment_iwslt14_3080x2_1024 m-B-3-1-H12-UR40M m-B-3-1-N-UR40M
pair_experiment_iwslt14_3080x2_1536_100k m-B-3-1-N-UR25M m-B-3-1-H12-UR25M 
