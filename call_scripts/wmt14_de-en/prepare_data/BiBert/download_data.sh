DATAPATH=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/bibert
DATASET=wmt-data.zip
mkdir -p $DATAPATH
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wbcnqwamiI5IfZrkNlQZmhvIsuSoCqwn' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wbcnqwamiI5IfZrkNlQZmhvIsuSoCqwn" -O $DATAPATH/$DATASET && rm -rf /tmp/cookies.txt
unzip $DATAPATH/$DATASET
