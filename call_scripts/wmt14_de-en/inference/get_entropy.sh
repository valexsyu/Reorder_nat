sed -n '/^E-.*/ p' < $1 > $2
sed -i 's/^E-.*\t//g' $2

BIBERT_DATA=/home/valexsyu/Doc/NMT/BiBERT/download_prepare/data/de-en-databin/generate-$3-entropy.out
BIBERT_DATA_CL=/home/valexsyu/Doc/NMT/BiBERT/download_prepare/data/de-en-databin/generate-$3-entropy_CL.out
sed -n '/^GGGGGGG-.*/ p' < $BIBERT_DATA > $BIBERT_DATA_CL
sed -i 's/^GGGGGGG-.* //g' $BIBERT_DATA_CL

HISTPATH="$(dirname "$2")"
python call_scripts/inference/get_entropy.py --input-path $2 --output-path $HISTPATH/  --bibert-path $BIBERT_DATA_CL