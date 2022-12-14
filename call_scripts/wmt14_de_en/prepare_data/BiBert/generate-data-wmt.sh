TGT=de
SRC=en
cd /home/valexsyu/Doc/NMT/BiBERT
source $HOME/.bashrc 
conda activate bibert

for trans_dir in "de-en" ; 
do
    if [ "$trans_dir" = "en-de" ]  
    then
        TGT=de
        SRC=en
    elif [ "$trans_dir" = "de-en" ]  
    then
        TGT=en
        SRC=de                            
    else        
        echo "error translation dir "
        exit 1
    fi    
    DATAPATH=./download_prepare_wmt_clean_deen/data_mixed_ft/
    STPATH=${DATAPATH}de-en-databin
    # STPATH=${DATAPATH}$SRC-$TGT-databin
    # MODELPATH=./models/dual-wmt-ft-$SRC$TGT/ 
    MODELPATH=./models/one-way-wmt-clean-de-en-12k/ 
    # MODELPATH=./models/dual-wmt/ 
    PRE_SRC=jhu-clsp/bibert-ende
    PRE=jhu-clsp/bibert-ende

    for prefix in "train" ; do 
        CUDA_VISIBLE_DEVICES=0 fairseq-generate \
        ${STPATH} --path ${MODELPATH}checkpoint_best.pt --bpe bert --pretrained_bpe ${PRE} --pretrained_bpe_src ${PRE_SRC} \
        --beam 4 --lenpen 0.6 --remove-bpe --vocab_file=${STPATH}/dict.en.txt \
        --gen-subset $prefix |tee ${STPATH}/generate-$prefix.out
    done
    #--batch-size 200 \

    DISTILLPATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/wmt14_clean_${SRC}_${TGT}_bibertDist_detoken
    mkdir -p $DISTILLPATH
    for prefix in "train" ; do 
        sed -n '/^S-.*/ p' < ${STPATH}/generate-$prefix.out > $DISTILLPATH/$prefix.$SRC
        sed -n '/^D-.*/ p' < ${STPATH}/generate-$prefix.out > $DISTILLPATH/$prefix.$TGT
    done

    for prefix in "train" ; do 
        sed -i 's/^S-.*\t//g' $DISTILLPATH/$prefix.$SRC
        sed -i 's/^D-.*\t//g' $DISTILLPATH/$prefix.$TGT
    done




    # for prefix in "valid" "test"; do 
    #     CUDA_VISIBLE_DEVICES=0 fairseq-generate \
    #     ${STPATH} --path ${MODELPATH}checkpoint_best.pt --bpe bert --pretrained_bpe ${PRE} --pretrained_bpe_src ${PRE_SRC} \
    #     --beam 4 --lenpen 0.6 --remove-bpe --vocab_file=${STPATH}/dict.en.txt \
    #     --gen-subset $prefix |tee ${STPATH}/generate-$prefix.out
    # done

    # DISTILLPATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/bibert/wmt14_$SRC-${TGT}_detoken_distill
    # mkdir -p $DISTILLPATH
    # for prefix in "valid"  "test"; do 
    #     sed -n '/^S-.*/ p' < ${STPATH}/generate-$prefix.out > $DISTILLPATH/$prefix.$SRC
    #     sed -n '/^T-.*/ p' < ${STPATH}/generate-$prefix.out > $DISTILLPATH/$prefix.$TGT
    # done

    # for prefix in "valid"  "test" ; do 
    #     sed -i 's/^S-.*\t//g' $DISTILLPATH/$prefix.$SRC
    #     sed -i 's/^T-.*\t//g' $DISTILLPATH/$prefix.$TGT
    # done    
done


