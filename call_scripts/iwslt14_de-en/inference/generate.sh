source $HOME/.bashrc 
conda activate base

# echo "============================================================================================"
# echo "                   Inference              gen                                               "
# echo "============================================================================================"
# bash call_scripts/iwslt14_de-en/inference/generate_ctc-reorder-battle-traindataset.sh
# echo "============================================================================================"
# echo "                   Inference              gen-1                                             "
# echo "============================================================================================"
# bash call_scripts/iwslt14_de-en/inference/generate_ctc-reorder-battle-traindataset-1.sh
# echo "============================================================================================"
# echo "                   Inference              gen-2                                            "
# echo "============================================================================================"
# bash call_scripts/iwslt14_de-en/inference/generate_ctc-reorder-battle-traindataset-2.sh
# echo "============================================================================================"
# echo "                   Inference              gen-3                                            "
# echo "============================================================================================"
# bash call_scripts/iwslt14_de-en/inference/generate_ctc-reorder-battle-traindataset-3.sh
# echo "============================================================================================"
# echo "                   Inference                gen-4                                            "
# echo "============================================================================================"
# bash call_scripts/iwslt14_de-en/inference/generate_ctc-reorder-battle-traindataset-4.sh
echo "============================================================================================"
echo "                   Inference                gen-6                                            "
echo "============================================================================================"
bash call_scripts/iwslt14_de-en/inference/generate_ctc-reorder-battle-traindataset-6.sh
echo "============================================================================================"
echo "                   Inference                gen-7                                            "
echo "============================================================================================"
bash call_scripts/iwslt14_de-en/inference/generate_ctc-reorder-battle-traindataset-7.sh
