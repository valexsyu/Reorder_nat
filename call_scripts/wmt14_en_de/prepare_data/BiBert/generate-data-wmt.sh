cd /home/valexsyu/Doc/NMT/BiBERT
source $HOME/.bashrc 
conda activate bibert

for trans_dir in "en-de" ; 
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
    DATAPATH=./download_prepare_wmt_clean_${SRC}${TGT}/data_mixed_ft/
    STPATH=${DATAPATH}de-en-databin
    MODELPATH=./models/one-way-wmt-clean-${SRC}-${TGT}-12k/  
    PRE_SRC=jhu-clsp/bibert-ende
    PRE=jhu-clsp/bibert-ende

    for prefix in "test" "train" ; do 
        CUDA_VISIBLE_DEVICES=0 fairseq-generate \
        ${STPATH} --path ${MODELPATH}checkpoint_best.pt --bpe bert --pretrained_bpe ${PRE} --pretrained_bpe_src ${PRE_SRC} \
        --beam 4 --lenpen 0.6 --remove-bpe --vocab_file=${STPATH}/dict.${TGT}.txt \
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
done


