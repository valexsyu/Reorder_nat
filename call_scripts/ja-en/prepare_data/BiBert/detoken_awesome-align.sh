DATAPATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/bibert/wmt-data
OUTDATAPATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/bibert/wmt14_detoken_awesome-align
## de-token data
mkdir $OUTDATAPATH
sed -r 's/( ##)//g' $DATAPATH/train.en > $OUTDATAPATH/train.en
sed -r 's/( ##)//g' $DATAPATH/train.de > $OUTDATAPATH/train.de
sed -r 's/( ##)//g' $DATAPATH/valid.en > $OUTDATAPATH/valid.en
sed -r 's/( ##)//g' $DATAPATH/valid.de > $OUTDATAPATH/valid.de
sed -r 's/( ##)//g' $DATAPATH/test.en > $OUTDATAPATH/test.en
sed -r 's/( ##)//g' $DATAPATH/test.de > $OUTDATAPATH/test.de
