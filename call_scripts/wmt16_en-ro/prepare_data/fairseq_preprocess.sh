#!/bin/bash
source $HOME/.bashrc 
conda activate bibert

# TEXT=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/baseline_token_distilled_wmt16_en_ro_mbert

# fairseq-preprocess --source-lang en --target-lang ro  --trainpref $TEXT/train --validpref $TEXT/valid \
# --testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
# --tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25 \

# TEXT=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/mbart_token_distilled_wmt16_en_ro_mbert

# fairseq-preprocess --source-lang en --target-lang ro  --trainpref $TEXT/train --validpref $TEXT/valid \
# --testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
# --tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25 \

# TEXT=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/wmt16_en_ro_token_mbert

# fairseq-preprocess --source-lang en --target-lang ro  --trainpref $TEXT/train --validpref $TEXT/valid \
# --testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
# --tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25 \



TEXT=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/wmt16_en_ro_BigBlDist_mbert

fairseq-preprocess --source-lang en --target-lang ro  --trainpref $TEXT/train --validpref $TEXT/valid \
--testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
--tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25 \


