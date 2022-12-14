#!/bin/bash
source $HOME/.bashrc 
conda activate bibert

# TEXT=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/baseline_token_distilled_wmt16_ro_en_mbert

# fairseq-preprocess --source-lang ro --target-lang en  --trainpref $TEXT/train --validpref $TEXT/valid \
# --testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
# --tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25 \

# TEXT=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/fnc_token_distilled_wmt16_ro_en_mbert

# fairseq-preprocess --source-lang ro --target-lang en  --trainpref $TEXT/train --validpref $TEXT/valid \
# --testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
# --tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25 \



# TEXT=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/wmt16_ro_en_token_mbert

# fairseq-preprocess --source-lang ro --target-lang en  --trainpref $TEXT/train --validpref $TEXT/valid \
# --testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
# --tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25 \


TEXT=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/wmt16_ro_en_BigBlDist_mbert

fairseq-preprocess --source-lang ro --target-lang en  --trainpref $TEXT/train --validpref $TEXT/valid \
--testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
--tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25 \