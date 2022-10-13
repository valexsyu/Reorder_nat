#!/bin/bash
#-----Setting-----#
SRC=de
TGT=en
FAIRSEQ_ROOT=/home/valex/Documents/Study/NMT/reorder/fairseq
BIBERT_ROOT=/home/valex/Documents/Study/NMT/reorder/BiBERT
AWESOME_ROOT=/home/valex/Documents/Study/NMT/awesome
BPEROOT=$FAIRSEQ_ROOT/examples/translation/subword-nmt/subword_nmt
DISTILLED_DATA_PATH=$FAIRSEQ_ROOT/examples/translation/data_$SRC-$TGT-distilled
DISTILLED_DUAL_DATA_PATH=$FAIRSEQ_ROOT/examples/translation/data_$SRC-$TGT-dual-distilled
DISTILLED_REORDER_DATA_PATH=$FAIRSEQ_ROOT/examples/translation/data_$SRC-$TGT-distilled_reordered
DISTILLED_REORDER_FOLDER=iwslt14.tokenized.$SRC-$TGT.distilled.reorder
DISTILLED_FOLDER=iwslt14.tokenized.$SRC-$TGT.distilled
DISTILLED_BLEU_FOLDER=iwslt14.tokenized.$SRC-$TGT.distilled.bleu
DISTILLED_DUAL_FOLDER=iwslt14.tokenized.$SRC-$TGT.dual-distilled
ORIGINAL_FOLDER=iwslt14.tokenized.de-en
ORIGINAL_DATA_PATH=$FAIRSEQ_ROOT/examples/translation/$ORIGINAL_FOLDER
ORIGINAL_DUAL_FOLDER=iwslt14.tokenized.de-en.dual
ORIGINAL_DUAL_DATA_PATH=$FAIRSEQ_ROOT/examples/translation/$ORIGINAL_DUAL_FOLDER
ORIGINAL_BLEU_SORTING_FOLDER=iwslt14.tokenized.de-en.bleu
ORIGINAL_BLEU_SORTING_DATA_PATH=$FAIRSEQ_ROOT/examples/translation/$ORIGINAL_BLEU_SORTING_FOLDER
REF_BLEU=$FAIRSEQ_ROOT/checkpoints/battle/No0-1/last.bleu/iwslt14.tokenized.de-en.train_data/generate-test.txt
REF_BLEU_MODEL=$FAIRSEQ_ROOT/checkpoints/battle/No0-1/checkpoint_last.pt
CURRICULAR_DATA_PATH=$FAIRSEQ_ROOT/examples/translation/data_curricular
CURRICULAR_FORDER=iwslt14.tokenized.$SRC-$TGT.distilled.reorder.curricular
CURRICULAR_DISTILLED_DATA_PATH=$FAIRSEQ_ROOT/examples/translation/data_curricular_$SRC-$TGT-distilled
CURRICULAR_DISTILLED_FORDER=iwslt14.tokenized.$SRC-$TGT.distilled.curricular
CURRICULAR_DUAL_DISTILLED_DATA_PATH=$FAIRSEQ_ROOT/examples/translation/data_curricular_$SRC-$TGT-dual-distilled
CURRICULAR_DUAL_DISTILLED_FORDER=iwslt14.tokenized.$SRC-$TGT.dual.distilled.curricular
AWESOME_MODEL_PATH=/home/valex/Documents/Study/NMT/reorder/AWESOME/awesome-align/output/train-finetune-5epoch


BIBERT_GEN="False"
REORDER_RUN="False"
BPE_RUN="True"
BLEU_SORTING="Fasle"
BIN_RUN="True"

TRANSLAT_DE="False"
DUAL_DISTILLED="False"
CUDA_DEVICES="0,1"
TOK_SIZE=10000
RUN_ALIGN_DATASET="train"
#-----------------------------------------
TEST="True"


GenerateDistilledData(){
    tmp=$BIBERT_ROOT/download_prepare/tmp_$SRC-$TGT
    if [ $TRANSLAT_DE == "False" ] ; then
        databin=$BIBERT_ROOT/download_prepare/data/$SRC-$TGT-databin
    else
        databin=$BIBERT_ROOT/download_prepare/data_mixed_ft_en_de/$SRC-$TGT-databin
    fi
    mkdir -p $tmp
    if [ ! -d $databin ] ; then
        echo "Error : databin path is not exist. path = $databin "
    fi
    # Copy train.bin to test.bin for AT model generate and use dict
    # echo "Message : Copy distill.train data"    
    for lang in $SRC $TGT; do
        for type in bin idx; do
            cp $databin/train.$SRC-$TGT.$lang.$type $tmp/test.$SRC-$TGT.$lang.$type
        done
        cp $databin/dict.$lang.txt $tmp/dict.$lang.txt
    done   

    cd $BIBERT_ROOT
    # Generate distilled data
    echo "Message : Generating Distilled Data"

    
    if [ $TRANSLAT_DE == "False" ] ; then
        STPATH=$tmp
        MODELPATH=./models/one-way/ 
        PRE_SRC=jhu-clsp/bibert-ende
        PRE=./download_prepare/8k-vocab-models
    else
        STPATH=$tmp
        MODELPATH=./models/dual-ft-$SRC-$TGT/ 
        PRE_SRC=jhu-clsp/bibert-ende
        PRE=./download_prepare/12k-vocab-models/      
    fi
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES fairseq-generate \
        ${STPATH} --path ${MODELPATH}checkpoint_best.pt --bpe bert --pretrained_bpe ${PRE} --pretrained_bpe_src ${PRE_SRC} \
        --beam 4 --lenpen 0.6 --remove-bpe --vocab_file=${STPATH}/dict.$TGT.txt \
        --max-len-a 1 --max-len-b 50|tee ${tmp}/generate.out    

    python train_data_from_generation_out.py --distilled $tmp/generate.out \
        --src-output $DISTILLED_DATA_PATH/train.$SRC-$TGT-distilled.$SRC --tgt-output $DISTILLED_DATA_PATH/train.$SRC-$TGT-distilled.$TGT

    cd $FAIRSEQ_ROOT/examples/translation
    SCRIPTS=mosesdecoder/scripts
    TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
    if [ $TRANSLAT_DE == "False" ] ; then
        sed -r 's/ '\'' /'\''/g' $DISTILLED_DATA_PATH/train.$SRC-$TGT-distilled.$TGT > $DISTILLED_DATA_PATH/train.$TGT.temp
        perl $TOKENIZER -threads 8 -l $TGT < $DISTILLED_DATA_PATH/train.$TGT.temp > $DISTILLED_DATA_PATH/train.$TGT
        rm $DISTILLED_DATA_PATH/train.$SRC-$TGT-distilled.$SRC
        rm $DISTILLED_DATA_PATH/train.$SRC-$TGT-distilled.$TGT
        rm $DISTILLED_DATA_PATH/train.$TGT.temp
    else
        sed -r 's/ '\'' /'\''/g' $DISTILLED_DATA_PATH/train.$SRC-$TGT-distilled.$TGT > $DISTILLED_DATA_PATH/train.$TGT.temp
        perl $TOKENIZER -threads 8 -l $TGT < $DISTILLED_DATA_PATH/train.$TGT.temp > $DISTILLED_DATA_PATH/train.$TGT
        #rm $DISTILLED_DATA_PATH/train.$SRC-$TGT-distilled.$TGT
        #rm $DISTILLED_DATA_PATH/train.$SRC-$TGT-distilled.$SRC
        rm $DISTILLED_DATA_PATH/train.$TGT.temp
        echo "$TGT word modify , ONLY DE NOW"
        python $FAIRSEQ_ROOT/call_scripts/reorder/de_word_modify.py --input-file $DISTILLED_DATA_PATH/train.$TGT --ref-file $ORIGINAL_DATA_PATH/tmp/train.$TGT
        mv $DISTILLED_DATA_PATH/train.modify.$TGT $DISTILLED_DATA_PATH/train.$TGT
    fi
    #rm -r $tmp
    cd $FAIRSEQ_ROOT
    
}

ReorderData(){
    echo "Message : Reordering ...."
    MODEL_NAME_OR_PATH=$1
    DATA_FILE=$2
    OUTPUT_PATH=$3 
    CONFIG_NAME=$MODEL_NAME_OR_PATH/config.json
    
    RUN_ALIGN_DATASET="train"
    if [ ! -d "$MODEL_NAME_OR_PATH" ]; then
        echo "ERROR : The AWENSOME MODEL is not exist. path : $MODEL_NAME_OR_PATH"
        exit 1
    fi
    #-----------Run Alignment---------------------#
    echo "Message : Run awsome-align"
    
    for f in $RUN_ALIGN_DATASET; do
        echo " run ${f} align ..."	          
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES awesome-align \
        --output_file=$OUTPUT_PATH/$f.align.$SRC-$TGT \
        --output_prob_file=$OUTPUT_PATH/$f.align_prob.$SRC-$TGT  \
        --model_name_or_path=$MODEL_NAME_OR_PATH \
        --data_file=$DATA_FILE \
        --extraction 'softmax' \
        --batch_size 200 
    done

    #----------------Run Max_bipartitle_match --------------------#
    echo "Message : run max_bipartitle_match"
    python call_scripts/reorder/max_bipartitle_match.py --root-path $OUTPUT_PATH \
        --data-type train
    mv $OUTPUT_PATH/train.$TGT.reorder $OUTPUT_PATH/train.$TGT

}

BPEProcess(){
    TRAIN=$1
    TOK_SIZE=$2
    INPUT_PATH=$3
    OUTPUT_PATH=$4
    echo "Message : learn_bpe.py on ${TRAIN}..."
    python $BPEROOT/learn_bpe.py -s $TOK_SIZE < $TRAIN > $OUTPUT_PATH/code  
    for L in $SRC $TGT; do
        for f in train.$L valid.$L test.$L; do
            echo "Message : apply_bpe.py to ${f}..."
            python $BPEROOT/apply_bpe.py -c $OUTPUT_PATH/code < $INPUT_PATH/$f > $OUTPUT_PATH/$f
        done
    done        
}

BleuSortingData(){
    echo "Message : Order the data set by BLEU using specific model"
    mkdir -p $2
    cp $1/* $2/
    generate_bleu="True"    
    checkpoint=checkpoints/battle/No0-1/checkpoint_last.pt
    data=iwslt14.tokenized.de-en.distilled.train_data #iwslt14.tokenized.$SRC-$TGT #Bin Data of TEST dataset 
    result_path=checkpoints/battle/No0-1/last.bleu/iwslt14.tokenized.de-en.distilled.train_data
    device=0
    # ---------------------------------------0
    if [ $generate_bleu == "True" ] ; then
        CUDA_VISIBLE_DEVICES=$device python generate.py \
            data-bin/$data \
            --gen-subset test \
            --task translation_lev \
            --path $checkpoint \
            --iter-decode-max-iter 0 \
            --iter-decode-eos-penalty 0 \
            --beam 1 --remove-bpe \
            --iter-decode-force-max-iter \
            --results-path $result_path \
            --arch nonautoregressive_transformer\
            --hist2d-only \
            --batch-size 150
    fi
    #BLEU high to low
    python call_scripts/reorder/sorting_train_data.py \
        --input-file-path $1 \
        --output-file-path $2 \
        --bleu-file $result_path/generate-test.txt \
        --src-lang $3 \
        --tgt-lang $4

    mv $2/train.sorting.$SRC $2/train.$SRC
    mv $2/train.sorting.$TGT $2/train.$TGT     
}

BINProcess(){
    DICT_PATH=$1
    INPUT_PATH=$2
    OUTPUT_PATH=$3   
    if [ -d $OUTPUT_PATH ] ; then
        rm -r $OUTPUT_PATH
    fi        
    echo "Message : generate Bin data. path : $OUTPUT_PATH"
    fairseq-preprocess --source-lang $SRC --target-lang $TGT \
        --trainpref $INPUT_PATH/train --validpref $INPUT_PATH/valid --testpref $INPUT_PATH/test \
        --destdir $OUTPUT_PATH \
        --bpe subword_nmt --tokenizer moses \
        --tgtdict $DICT_PATH/dict.$TGT.txt \
        --srcdict $DICT_PATH/dict.$SRC.txt \
        --workers 20       
}


#=================main==================================
#========================================================
#========================================================
if [ $BIBERT_GEN == "True" ] ; then
    mkdir -p $DISTILLED_DATA_PATH
    for prefix in "valid" "test" "train" ;
    do    
        for lang in $SRC $TGT ; 
        do
            cp $ORIGINAL_DATA_PATH/tmp/$prefix.$lang $DISTILLED_DATA_PATH
        done
    done
    GenerateDistilledData
    echo "Message : distill data is generated by trained BiBert"
else
    echo "Message : Not to generate distill data"
fi



if [ $REORDER_RUN == "True" ] ; then
    mkdir -p $DISTILLED_REORDER_DATA_PATH
    for prefix in "valid" "test" "train" ;
    do    
        for lang in $SRC $TGT ; 
        do
            cp $ORIGINAL_DATA_PATH/tmp/$prefix.$lang $DISTILLED_REORDER_DATA_PATH
        done
    done
    cp $DISTILLED_DATA_PATH/train.$TGT $DISTILLED_REORDER_DATA_PATH/train.$TGT
    paste -d' ||| ' $DISTILLED_DATA_PATH/train.$TGT /dev/null /dev/null /dev/null /dev/null  $DISTILLED_DATA_PATH/train.$TGT  > $DISTILLED_DATA_PATH/train.paste.$SRC-$TGT
    ReorderData $AWESOME_MODEL_PATH $DISTILLED_DATA_PATH/train.paste.$SRC-$TGT $DISTILLED_REORDER_DATA_PATH
    rm $DISTILLED_DATA_PATH/train.paste.$SRC-$TGT
else
    echo "Message : Not to RUN reorder process"
fi
   
if [ $BPE_RUN == "True" ] ; then
    echo "Message : BPE_RUN"
    tmp=$FAIRSEQ_ROOT/bpe_temp 
    mkdir $tmp
    bpe_learn_data=$tmp/train.$SRC-$TGT.all
    for prefix in "valid" "test" "train" ;
    do    
        for lang in $SRC $TGT ; 
        do 
            cat $ORIGINAL_DATA_PATH/tmp/$prefix.$lang >> $bpe_learn_data
        done    
    done 
    cat $DISTILLED_DATA_PATH/train.$TGT >> $bpe_learn_data
    
    if [ $DUAL_DISTILLED == "True" ] ; then
        cat $DISTILLED_DUAL_DATA_PATH/train.$SRC >> $bpe_learn_data  
        output_folder=$DISTILLED_DUAL_DATA_PATH/$DISTILLED_DUAL_FOLDER
        input_folder=$DISTILLED_DUAL_DATA_PATH
        mkdir $output_folder
        BPEProcess $bpe_learn_data $TOK_SIZE $input_folder $output_folder  

        output_folder=$ORIGINAL_DUAL_DATA_PATH/
        input_folder=$ORIGINAL_DATA_PATH/tmp
        BPEProcess $bpe_learn_data $TOK_SIZE $input_folder $output_folder       
    else
        output_folder=$DISTILLED_REORDER_DATA_PATH/$DISTILLED_REORDER_FOLDER
        input_folder=$DISTILLED_REORDER_DATA_PATH
        mkdir $output_folder
        BPEProcess $bpe_learn_data $TOK_SIZE $input_folder $output_folder

        output_folder=$DISTILLED_DATA_PATH/$DISTILLED_FOLDER
        input_folder=$DISTILLED_DATA_PATH
        mkdir $output_folder
        BPEProcess $bpe_learn_data $TOK_SIZE $input_folder $output_folder

        output_folder=$ORIGINAL_DATA_PATH/
        input_folder=$ORIGINAL_DATA_PATH/tmp
        BPEProcess $bpe_learn_data $TOK_SIZE $input_folder $output_folder
    fi
    rm -r $tmp
else
    echo "Message : Not to run BPE_RUN"
fi


if [ $BLEU_SORTING == "True" ] ; then
    cd $FAIRSEQ_ROOT
    echo "Message : Run BLEU Sorting"
    #BleuSortingData $ORIGINAL_DUAL_DATA_PATH $ORIGINAL_BLEU_SORTING_DATA_PATH $SRC $TGT
    #BleuSortingData $ORIGINAL_DATA_PATH $ORIGINAL_BLEU_SORTING_DATA_PATH $SRC $TGT 
    BleuSortingData $DISTILLED_DATA_PATH/$DISTILLED_FOLDER $DISTILLED_DATA_PATH/$DISTILLED_BLEU_FOLDER $SRC $TGT 
    cd -
else
    echo "Message : Not to Run BLEU Sorting"
fi    

MakeCurricularData(){
    mkdir -p $1
    cp $2/* $1/
    cat $3/train.$TGT >> $1/train.$TGT
    cat $3/train.$SRC >> $1/train.$SRC    
}

if [ $BIN_RUN == "True" ] ; then
    echo "Message : BIN_RUN"
    #--------------------------all dict------------------------------
    tmp=$FAIRSEQ_ROOT/bin_temp
    mkdir $tmp
    dictionary_already_exist="True"
    if [ $dictionary_already_exist == "False" ] ; then
        for prefix in "valid" "test" "train" ;
        do    
            for lang in $SRC $TGT ; 
            do
                if [ $DUAL_DISTILLED == "True" ] ; then
                    cat $DISTILLED_DUAL_DATA_PATH/$DISTILLED_DUAL_FOLDER/$prefix.$lang >> $tmp/train.all.$lang
                    if [ $prefix == "train" ] ; then
                        cat $ORIGINAL_DUAL_DATA_PATH/$prefix.$lang >> $tmp/train.all.$lang
                    fi                     
                else
                    cat $DISTILLED_REORDER_DATA_PATH/$DISTILLED_REORDER_FOLDER/$prefix.$lang >> $tmp/train.all.$lang
                    if [ $prefix == "train" ] ; then
                        cat $ORIGINAL_DATA_PATH/$prefix.$lang >> $tmp/train.all.$lang
                    fi                
                fi

            done    
        done            
        fairseq-preprocess --source-lang $SRC --target-lang $TGT \
            --trainpref $tmp/train.all --validpref $ORIGINAL_DATA_PATH/valid --testpref $ORIGINAL_DATA_PATH/test \
            --destdir $tmp \
            --bpe subword_nmt --tokenizer moses \
            --joined-dictionary \
            --workers 20  
    else
        cp $FAIRSEQ_ROOT/data-bin/$ORIGINAL_FOLDER/* $tmp
    fi         

    if [ $DUAL_DISTILLED == "True" ] ; then
        #BINProcess $tmp $DISTILLED_DUAL_DATA_PATH/$DISTILLED_DUAL_FOLDER $FAIRSEQ_ROOT/data-bin/$DISTILLED_DUAL_FOLDER
        #BINProcess $tmp $ORIGINAL_DUAL_DATA_PATH $FAIRSEQ_ROOT/data-bin/$ORIGINAL_DUAL_FOLDER
        MakeCurricularData $CURRICULAR_DUAL_DISTILLED_DATA_PATH/$CURRICULAR_DUAL_DISTILLED_FORDER $DISTILLED_DUAL_DATA_PATH/$DISTILLED_DUAL_FOLDER  $ORIGINAL_BLEU_SORTING_DATA_PATH
        BINProcess $tmp $CURRICULAR_DUAL_DISTILLED_DATA_PATH/$CURRICULAR_DUAL_DISTILLED_FORDER $FAIRSEQ_ROOT/data-bin/$CURRICULAR_DUAL_DISTILLED_FORDER             
    else
        BINProcess $tmp $DISTILLED_REORDER_DATA_PATH/$DISTILLED_REORDER_FOLDER $FAIRSEQ_ROOT/data-bin/$DISTILLED_REORDER_FOLDER
        BINProcess $tmp $DISTILLED_DATA_PATH/$DISTILLED_FOLDER $FAIRSEQ_ROOT/data-bin/$DISTILLED_FOLDER
        BINProcess $tmp $ORIGINAL_DATA_PATH $FAIRSEQ_ROOT/data-bin/$ORIGINAL_FOLDER
        MakeCurricularData $CURRICULAR_DATA_PATH/$CURRICULAR_FORDER $DISTILLED_REORDER_DATA_PATH/$DISTILLED_REORDER_FOLDER $ORIGINAL_DATA_PATH
        BINProcess $tmp $CURRICULAR_DATA_PATH/$CURRICULAR_FORDER $FAIRSEQ_ROOT/data-bin/$CURRICULAR_FORDER
        #MakeCurricularData $CURRICULAR_DISTILLED_FORDER/$CURRICULAR_DISTILLED_FORDER $DISTILLED_DATA_PATH/$DISTILLED_BLEU_FOLDER $ORIGINAL_BLEU_SORTING_DATA_PATH
        #BINProcess $tmp $CURRICULAR_DISTILLED_FORDER/$CURRICULAR_DISTILLED_FORDER $FAIRSEQ_ROOT/data-bin/$CURRICULAR_DISTILLED_FORDER

    fi

    ##rm -r $tmp
else
    echo "Message : Not to run BIN_RUN"    
fi


:<<'EndComment'






EndComment









