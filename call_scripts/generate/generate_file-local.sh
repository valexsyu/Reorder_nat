source $HOME/.bashrc 
conda activate base

#--skip-exist-genfile --load-exist-bleu 
CUDA_VISIBLE_DEVICES=1 bash call_scripts/generate_nat.sh -b 50 --local --data-subset valid --ck-types top --avg-speed 1 --no-atten-mask --avg-ck-turnoff \
-e 2-2-1-1-N-UF20T \
-e 2-2-1-1-H12-UF20T \
-e 2-2-1-1-H11-UF20T \
-e 2-2-1-1-H10-UF20T \
-e 2-2-1-1-H9-UF20T \
-e 2-2-1-1-H8-UF20T \
-e 2-2-1-1-H7-UF20T \
-e 2-2-1-1-H6-UF20T \
-e 2-2-1-1-H5-UF20T \
-e 2-2-1-1-H4-UF20T \
-e 2-2-1-1-H3-UF20T \
-e 2-2-1-1-H2-UF20T \
-e 2-2-1-1-H1-UF20T \
-e 2-2-1-1-N-UF20M   \
-e 2-2-1-1-H12-UF20M \
-e 2-2-1-1-H11-UF20M \
-e 2-2-1-1-H10-UF20M \
-e 2-2-1-1-H9-UF20M  \
-e 2-2-1-1-H8-UF20M  \
-e 2-2-1-1-H7-UF20M  \
-e 2-2-1-1-H6-UF20M  \
-e 2-2-1-1-H5-UF20M  \
-e 2-2-1-1-H4-UF20M  \
-e 2-2-1-1-H3-UF20M  \
-e 2-2-1-1-H2-UF20M  \
-e 2-2-1-1-H1-UF20M  \
-e 1-1-1-1-N-UF20T   \
-e 1-1-1-1-H12-UF20T \
-e 1-1-1-1-H11-UF20T \
-e 1-1-1-1-H10-UF20T \
-e 1-1-1-1-H9-UF20T  \
-e 1-1-1-1-H8-UF20T  \
-e 1-1-1-1-H7-UF20T  \
-e 1-1-1-1-H6-UF20T  \
-e 1-1-1-1-H5-UF20T  \
-e 1-1-1-1-H4-UF20T  \
-e 1-1-1-1-H3-UF20T  \
-e 1-1-1-1-H2-UF20T  \
-e 1-1-1-1-H1-UF20T  \
-e 1-1-1-1-N-UF20M  \
-e 1-1-1-1-H12-UF20M \
-e 1-1-1-1-H11-UF20M \
-e 1-1-1-1-H10-UF20M \
-e 1-1-1-1-H9-UF20M  \
-e 1-1-1-1-H8-UF20M  \
-e 1-1-1-1-H7-UF20M  \
-e 1-1-1-1-H6-UF20M  \
-e 1-1-1-1-H5-UF20M  \
-e 1-1-1-1-H4-UF20M  \
-e 1-1-1-1-H3-UF20M  \
-e 1-1-1-1-H2-UF20M  \
-e 1-1-1-1-H1-UF20M  



        # 2-2-1-1-H5-UF20T
        # 2-2-1-1-H4-UF20T
        # 2-2-1-1-H3-UF20T
        # 2-2-1-1-H2-UF20T
        # 2-2-1-1-H1-UF20T
        # 2-2-1-1-H5-UF20M
        # 2-2-1-1-H4-UF20M
        # 2-2-1-1-H3-UF20M
        # 2-2-1-1-H2-UF20M
        # 2-2-1-1-H1-UF20M
        # 1-1-1-1-N-UF20T
        # 1-1-1-1-H12-UF20T
        # 1-1-1-1-H10-UF20T
        # 1-1-1-1-H8-UF20T
        # 1-1-1-1-H5-UF20T
        # 1-1-1-1-H4-UF20T
        # 1-1-1-1-H3-UF20T
        # 1-1-1-1-H2-UF20T
        # 1-1-1-1-H1-UF20T
        # 1-1-1-1-H8-UF20M
        # 1-1-1-1-H4-UF20M
        # 1-1-1-1-H3-UF20M
        # 1-1-1-1-H2-UF20M
        # 1-1-1-1-H1-UF20M