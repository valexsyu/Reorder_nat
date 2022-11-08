TGT=de
SRC=en
cd /home/valexsyu/Doc/NMT/BiBERT
source $HOME/.bashrc 
conda activate bibert

for trans_dir in "de-en" ;  #"en-de" "de-en"
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
    DATAPATH=/home/valexsyu/Doc/NMT/BiBERT/download_prepare/wmt-data/
    STPATH=${DATAPATH}$SRC-$TGT-databin
    MODELPATH=./models/dual-wmt-ft-$SRC$TGT/ 
    PRE_SRC=jhu-clsp/bibert-ende
    PRE=jhu-clsp/bibert-ende

    for prefix in "train" ; do 
        CUDA_VISIBLE_DEVICES=0 fairseq-generate \
        ${STPATH} --path ${MODELPATH}checkpoint_best.pt --bpe bert --pretrained_bpe ${PRE} --pretrained_bpe_src ${PRE_SRC} \
        --beam 1 --lenpen 0.6 --remove-bpe --vocab_file=${STPATH}/dict.en.txt --batch-size 200 \
        --gen-subset $prefix |tee ${STPATH}/generate-nondistill-$prefix.out
    done

    DISTILLPATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/bibert/wmt14_$SRC-${TGT}_detoken_nondistill
    mkdir -p $DISTILLPATH
    for prefix in "train" ; do 
        sed -n '/^S-.*/ p' < ${STPATH}/generate-nondistill-$prefix.out > $DISTILLPATH/$prefix.$SRC
        sed -n '/^T-.*/ p' < ${STPATH}/generate-nondistill-$prefix.out > $DISTILLPATH/$prefix.$TGT
    done

    for prefix in "train" ; do 
        sed -i 's/^S-.*\t//g' $DISTILLPATH/$prefix.$SRC
        sed -i 's/^T-.*\t//g' $DISTILLPATH/$prefix.$TGT
    done
done


