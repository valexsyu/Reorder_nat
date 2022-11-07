# ## prepare bibert data

cd /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/wmt16_en-ro/prepare_data/mBert
bash download_and_prepare_data.sh 

cd /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/wmt16_en-ro/prepare_data/
bash fairseq_preprocess.sh
