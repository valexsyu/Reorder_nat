
# bash call_scripts/tool/cal_diff_score_to_get_max_bleu/generate_file_rename.sh r-E-3-1-N-UR

prefix_nam=K-2-3-1-N-UR

mkdir checkpoints/sel_rate/${prefix_nam}XXM
cp  checkpoints/${prefix_nam}20M/test/best_top5_10_1.bleu/generate-test.txt checkpoints/sel_rate/${prefix_nam}XXM/generate-test-2.0.txt
cp  checkpoints/${prefix_nam}30M/test/best_top5_10_1.bleu/generate-test.txt checkpoints/sel_rate/${prefix_nam}XXM/generate-test-3.0.txt
cp  checkpoints/${prefix_nam}40M/test/best_top5_10_1.bleu/generate-test.txt checkpoints/sel_rate/${prefix_nam}XXM/generate-test-4.0.txt