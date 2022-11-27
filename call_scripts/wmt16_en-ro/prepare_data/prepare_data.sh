# ## prepare bibert data

cd /home/valex/Documents/Study/battleship/Reorder_nat/call_scripts/wmt16_en-ro/prepare_data/mBert
bash download_and_prepare_data.sh 

cd /home/valex/Documents/Study/battleship/Reorder_nat/call_scripts/wmt16_en-ro/prepare_data
bash fairseq_preprocess.sh
