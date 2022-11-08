# ## prepare bibert data
source $HOME/.bashrc 
conda activate base
bash /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/wmt14_de_en/prepare_data/BiBert/download_and_prepare_data.sh
## fairseq preprocess
bash /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/wmt14_de_en/prepare_data/prepare_data.sh
