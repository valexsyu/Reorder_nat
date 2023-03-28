# MODEL_NAME=bert-base-multilingual-uncased
INPUT_PATH=data/nat_position_reorder/awesome/iwslt14_de_en_detoken
# echo 'Cloning Moses github repository (for tokenization scripts)...'
# git clone https://github.com/moses-smt/mosesdecoder.gits
SCRIPTS=call_scripts/tool/mosesdecoder/scripts


## de-mose data
mkdir $INPUT_PATH/demose
for prefix in "train" "valid" "test" ;
do
    for lang in "en" "de" ;
    do
        echo "demose ${prefix}.${lang} "
        perl $SCRIPTS/tokenizer/detokenizer.perl -l en -q < $INPUT_PATH/${prefix}.${lang} > $INPUT_PATH/demose/${prefix}.${lang}
    done
done