#!/bin/bash
source $HOME/.bashrc 
conda activate bibert

bash /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/iwslt14_en-de/prepare_data/Bibert/download_and_prepare_data.sh
bash /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/iwslt14_en-de/prepare_data/fairseq_preprocess.sh

