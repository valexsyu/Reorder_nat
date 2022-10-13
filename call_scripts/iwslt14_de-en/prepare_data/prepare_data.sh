# ## prepare bibert data
  
# conda activate bibert
#conda init bash 
#conda activate bibert
#cd /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/prepare_data/BiBert
#bash download_and_prepare_data.sh 
#bash detoken_awesome-align.sh

# # distill data from bibert
# cd ..
# bash train.sh 
#bash generate-data.sh 
#bash distill_data_tokenzation.sh

# #awesome
# cd /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/prepare_data/awesome
#conda init bash
#conda activate awesome
# bash paste_srctgt.sh
# cd /home/valexsyu/Doc/NMT/awesome-align
# bash train_iwslt14_deen_battle_mbert.sh
# bash scrips/align_iwlst14_ende_finetune-battle.sh
# cd /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/prepare_data/awesome
# bash complete_align.sh

## fairseq preprocess
#conda init bash  
#conda activate bibert
cd /home/valexsyu/Doc/NMT/Reorder_nat/call_scripts/prepare_data
bash fairseq_preprocess.sh
