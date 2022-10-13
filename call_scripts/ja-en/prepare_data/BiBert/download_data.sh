
# #==========================wmt-data================================
# DATAPATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/bibert
# DATASET=wmt20-jaen-data.zip
# mkdir -p $DATAPATH
# # wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wbcnqwamiI5IfZrkNlQZmhvIsuSoCqwn' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wbcnqwamiI5IfZrkNlQZmhvIsuSoCqwn" -O $DATAPATH/$DATASET && rm -rf /tmp/cookies.txt
# wget https://dl.fbaipublicfiles.com/nat/fully_nat/datasets/wmt20.ja-en.zip -O $DATAPATH/$DATASET
# unzip $DATAPATH/$DATASET

DATAPATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/bibert/wmt20_jaen_filter09995/ja-en.distill
OUTPATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/bibert/wmt20_jaen_distill
mkdir -p $OUTPATH
sed -e 's/ //g' -e 's/▁/ /g' -e 's/ //' $DATAPATH/distill.ja-en.en > $OUTPATH/train.en
sed -e 's/ //g' -e 's/▁/ /g' -e 's/ //' $DATAPATH/distill.ja-en.ja > $OUTPATH/train.ja


OUTPATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/bibert/wmt20_jaen
mkdir -p $OUTPATH
cp $/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/bibert/wmt20_jaen_distill/train.ja $OUTPATH/train.ja
sed -e 's/ //g' -e 's/▁/ /g' -e 's/ //' target.ja-en.en > train.en 

