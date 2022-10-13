# ## prepare bibert data
  
# conda activate bibert
#conda init bash 
#conda activate bibert
# cd /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/wmt14_de-en/prepare_data/BiBert
# bash download_data.sh 
# bash detoken_awesome-align.sh

# # distill data from bibert
# cd /home/valexsyu/Doc/NMT/BiBERT
# bash train-wmt-de-en.sh 
# bash generate-wmt-data.sh 
# bash distill_data_tokenzation.sh

# #awesome
# cd /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/prepare_data/awesome
#conda init bash
#conda activate awesome
# bash paste_srctgt.sh
# cd /home/valexsyu/Doc/NMT/awesome-align
# bash train_wmt14_deen_battle_mbert.sh
# bash scrips/align_wmt14_ende_finetune-battle.sh
# cd /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/prepare_data/awesome
# bash complete_align.sh

## fairseq preprocess
#conda init bash  
#conda activate bibert
# cd /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/prepare_data
# bash fairseq_preprocess.sh
