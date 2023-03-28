cat generate-test.txt |grep -P "^T-" | sort -V | cut -f 2-  > test_sort.en
cat generate-test.txt |grep -P "^D" | sort -V  | cut -f 3- > hopy_sort.en