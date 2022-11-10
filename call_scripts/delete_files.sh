
for dir in B-1-1-1-H12-UF30M;
do
for i in 1 2 3 4 5 6;
do 
    rm $dir/checkpoint_*_$i*.pt

done

for i in 8 9;
do 
    rm $dir/checkpoint_*_$i*.pt

done

# rm $dir/checkpoint.best_bleu_*

# mv $dir /livingrooms/jcx/cowork_data/checkpoints_dir/

done