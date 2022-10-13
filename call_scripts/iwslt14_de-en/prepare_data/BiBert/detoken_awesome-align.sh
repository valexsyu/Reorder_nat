DATAPATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/bibert
## de-token data
mkdir detoken_awesome-align
sed -r 's/( ##)//g' data/train.en > detoken_awesome-align/train.en
sed -r 's/( ##)//g' data/train.de > detoken_awesome-align/train.de
sed -r 's/( ##)//g' data/valid.en > detoken_awesome-align/valid.en
sed -r 's/( ##)//g' data/valid.de > detoken_awesome-align/valid.de
sed -r 's/( ##)//g' data/test.en > detoken_awesome-align/test.en
sed -r 's/( ##)//g' data/test.de > detoken_awesome-align/test.de

mkdir -p $DATAPATH
mv 8k-vocab-models 12k-vocab-models data data_mixed data_mixed_ft detoken_awesome-align $DATAPATH