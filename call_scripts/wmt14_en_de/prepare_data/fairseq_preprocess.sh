#!/bin/bash
source $HOME/.bashrc 
conda activate bibert

# TEXT=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_tokenize_wmt14_de_en

# fairseq-preprocess --source-lang de --target-lang en  --trainpref $TEXT/train --validpref $TEXT/valid \
# --testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
# --tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25 --align-suffix align \

# TEXT=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_token_distill_wmt14_de_en

# fairseq-preprocess --source-lang de --target-lang en  --trainpref $TEXT/train --validpref $TEXT/valid \
# --testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
# --tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25 --align-suffix align \


# TEXT=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_token_distill_wmt14_en_de_52k

# fairseq-preprocess --source-lang en --target-lang de  --trainpref $TEXT/train --validpref $TEXT/valid \
# --testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
# --tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25 --align-suffix align \

# TEXT=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_token_distill_wmt14_de_en_mbert

# fairseq-preprocess --source-lang de --target-lang en  --trainpref $TEXT/train --validpref $TEXT/valid \
# --testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
# --tgtdict $TEXT/src_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25 --align-suffix align \


# TEXT=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/Bibert_token_distill_wmt14_de_en_dual-bert-de-en

# fairseq-preprocess --source-lang de --target-lang en  --trainpref $TEXT/train --validpref $TEXT/valid \
# --testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
# --tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25 --align-suffix align \


# TEXT=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/baseline_token_distilled_wmt14_en_de_bibert

# fairseq-preprocess --source-lang en --target-lang de  --trainpref $TEXT/train --validpref $TEXT/valid \
# --testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
# --tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25 --align-suffix align \

# TEXT=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/BigBl_token_distill_wmt14_en_de_bibert

# fairseq-preprocess --source-lang en --target-lang de  --trainpref $TEXT/train --validpref $TEXT/valid \
# --testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
# --tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25 --align-suffix align \

# TEXT=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/wmt14_clean_en_de_token_bibert

# fairseq-preprocess --source-lang en --target-lang de  --trainpref $TEXT/train --validpref $TEXT/valid \
# --testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
# --tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25 --align-suffix align \


# TEXT=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/wmt14_clean_en_de_bibertDist_bibert

# fairseq-preprocess --source-lang en --target-lang de  --trainpref $TEXT/train --validpref $TEXT/valid \
# --testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
# --tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25 --align-suffix align \


TEXT=/home/valexsyu/Doc/NMT/Reorder_nat/data/nat_position_reorder/awesome/wmt14_clean_en_de_6kval_cased

fairseq-preprocess --source-lang en --target-lang de  --trainpref $TEXT/train --validpref $TEXT/valid \
--testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
--tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25 --align-suffix align \