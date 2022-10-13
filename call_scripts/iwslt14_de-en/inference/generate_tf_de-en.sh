source $HOME/.bashrc 
conda activate base
fairseq-generate data/nat_position_reorder/awesome/Bibert_token_iwslt14_de_en_52k/de-en-databin \
    --path checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 1 --remove-bpe