#!/bin/bash
source $HOME/.bashrc 
conda activate bibert

cd Bibert/
bash download_and_prepare_data.sh 
cd ../
bash fairseq_preprocess.sh
