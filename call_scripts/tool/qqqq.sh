# mkdir z-2-3-1-H12-UF40T-50k-fixpos-5
# mkdir z-2-4-1-H12-UF40T-50k-fixpos-5
# mkdir z-2-3-1-H12-UF40T-50k-5
# mkdir z-2-3-1-H12-UF40M-50k-5
# mkdir z-2-3-1-H12-UR40T-50k-5
# mkdir z-2-3-1-H12-UR40M-50k-5

# cp 2-2-3-1-H12-UF40T-50k-fixpos-5/checkpoint_best_top5.pt z-2-3-1-H12-UF40T-50k-fixpos-5
# cp 2-2-4-1-H12-UF40T-50k-fixpos-5/checkpoint_best_top5.pt z-2-4-1-H12-UF40T-50k-fixpos-5
# cp 2-2-3-1-H12-UF40T-50k-5/checkpoint_best_top5.pt z-2-3-1-H12-UF40T-50k-5
# cp 2-2-3-1-H12-UF40M-50k-5/checkpoint_best_top5.pt z-2-3-1-H12-UF40M-50k-5
# cp 2-2-3-1-H12-UR40T-50k-5/checkpoint_best_top5.pt z-2-3-1-H12-UR40T-50k-5
# cp 2-2-3-1-H12-UR40M-50k-5/checkpoint_best_top5.pt z-2-3-1-H12-UR40M-50k-5

data_inds=("12411" "58550" "44847" "103105" "72031" "134598" "217907" "251200" "153523" "406432")
data_from=data/nat_position_reorder/awesome/wmt14_clean_de_en_6kval_bibert
data_to=data/nat_position_reorder/awesome/iwslt14_de_en_bibertDist_bibert_addlongsent
for data_idx in "${data_inds[@]}"; do
    sed -n "${data_idx}p" $data_from/train.en >> $data_to/test.en
    sed -n "${data_idx}p" $data_from/train.de >> $data_to/test.de
done

