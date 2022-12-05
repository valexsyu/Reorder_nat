#!/bin/bash
source $HOME/.bashrc 
conda activate bibert


# #
# TEXT=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/iwslt14_en_de_token_bibert

# fairseq-preprocess --source-lang en --target-lang de  --trainpref $TEXT/train --validpref $TEXT/valid \
# --testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
# --tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25 --align-suffix align \


TEXT=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/baseline_token_distill_iwslt14_en_de_bibert

fairseq-preprocess --source-lang en --target-lang de  --trainpref $TEXT/train --validpref $TEXT/valid \
--testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
--tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25 --align-suffix align \

