source $HOME/.bashrc 
conda activate base

source call_scripts/train/pair_experiment.sh

# pair_experiment_iwslt14_3080x2_1024 m-B-3-1-H12-UR40M m-B-3-1-N-UR40M
# pair_experiment_iwslt14_3080x2_1536_100k m-B-3-1-N-UR25M m-B-3-1-H12-UR25M 

pair_experiment_iwslt14_4_1024_50k m-B-3-1-H12-UR25M-50k
pair_experiment_iwslt14_4_1024_50k m-B-3-1-H12-UR30M-50k
pair_experiment_iwslt14_4_1024_50k m-B-3-1-H12-UR20M-50k



