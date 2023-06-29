# e.g. 
# bash call_scripts/tool/cal_diff_score_to_get_max_bleu/Find_upper_bound/script.sh \
#       -d generate-test-2.0.txt -d generate-test-3.0.txt -d generate-test-4.0.txt \
#       -t generate-test-2.0.txt -p sel_rate/m-B-1-1-N-URXXM -e m
# bash call_scripts/tool/cal_diff_score_to_get_max_bleu/Find_upper_bound/script.sh \
#       -d generate-test-2.0.txt -d generate-test-3.0.txt -d generate-test-4.0.txt \
#       -t generate-test-2.0.txt -p sel_rate/out.wmt14.de-en.alg1 -e Fail


function default_setting() { 
    root_path=/home/valexsyu/Doc/NMT/Reorder_nat
}


default_setting

VALID_ARGS=$(getopt -o d: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1;
fi

eval set -- "$VALID_ARGS"

while [ : ]; do
    case "$1" in 
        -d )
            data_path="$2"       
            shift 2
        ;;                   
        --) shift; 
            break    
    esac
done

echo "========================================================"

file_path=$root_path/$data_path
folderpath=$(dirname "$file_path")



cat $file_path | head -n -2 | grep -P "^T" |sort -V |cut -f 2-  > $folderpath/T-sentence.txt
cat $file_path | head -n -2 | grep -P "^D" | sort -V | cut -f 3-  > $folderpath/D-sentence.txt

python call_scripts/tool/generate_file_bleu/corpus_bleu.py \
            --ref-path $folderpath/T-sentence.txt \
            --hypo-path $folderpath/D-sentence.txt \


perl call_scripts/tool/cal_diff_score_to_get_max_bleu/multi-bleu.perl $folderpath/T-sentence.txt  < $folderpath/D-sentence.txt
