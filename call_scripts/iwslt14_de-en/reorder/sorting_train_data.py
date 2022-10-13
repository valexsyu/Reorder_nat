# coding=utf-8

import argparse
from tqdm import tqdm 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file-path', type=str, default="train.en",  help='path or name of the input data')
    parser.add_argument('--bleu-file', type=str, default="generate-test.txt",  help='path or name of the bleu data')
    parser.add_argument('--output-file-path', type=str, default="train.sorting.en" ,  help='path or name of output data')
    parser.add_argument('--src-lang', type=str, default="de" ,  help='source language')
    parser.add_argument('--tgt-lang', type=str, default="en" ,  help='target language')
    args = parser.parse_args()  
    
    input_src_path = args.input_file_path + "/train." + args.src_lang
    input_tgt_path = args.input_file_path + "/train." + args.tgt_lang
    output_src_path = args.output_file_path + "/train.sorting." + args.src_lang
    output_tgt_path = args.output_file_path + "/train.sorting." + args.tgt_lang
    with open(input_src_path, "r", encoding="utf-8") as f:
        input_src_lines = f.readlines()
    with open(input_tgt_path, "r", encoding="utf-8") as f:
        input_tgt_lines = f.readlines()        
    with open(args.bleu_file, "r", encoding="utf-8") as f:
        bleu_lines = f.readlines()        
    with open(output_src_path, "w", buffering=1, encoding="utf-8") as f:
        with open(output_tgt_path, "w", buffering=1, encoding="utf-8") as g:
            sorting(input_src_lines, input_tgt_lines, bleu_lines, f, g)
        

def sorting(src_lines, tgt_lines, bleu_lines, out_src_file, out_tgt_file):
    data_lines = list(zip(src_lines, tgt_lines))
    bleu_lines_list = []
    for bleu_line in (bleu_lines[:-1]):
        bleu_line = (bleu_line.replace('\t'," ").replace("\n","")).strip(" ").split(" ")
        bleu_line[0] = int(bleu_line[0])
        bleu_line[1] = float(bleu_line[1])
        bleu_lines_list.append(bleu_line)
    bleu_lines_list = sorted(bleu_lines_list , key=lambda line : line[0])
    #bleu_lines_list = [[0 , 10, "112"], [1 , 23, "112"],[2 , 30, "112"],[3 , 2, "112"]]
    #data_lines = [('1 1 1', '1 1 1'), ('1 2', '1 2'),('1 1 3', '1 1 3'),('1 1 4', '1 1 4')]
    bleu_list =[i[1] for i in bleu_lines_list]
    data_lines_sorted_index = sorted(range(len(bleu_lines_list)) , key=lambda line : bleu_list[line], reverse=True)
    data_lines_bleu = [data_lines[i] for i in data_lines_sorted_index ]
    for _, data_line_bleu in enumerate(data_lines_bleu):
        src_line, tgt_line = data_line_bleu[0], data_line_bleu[1]   
        src_line = src_line.replace('\t'," ").replace("\n","")
        tgt_line = tgt_line.replace('\t'," ").replace("\n","")
        print(src_line , file=out_src_file)
        print(tgt_line , file=out_tgt_file)
        
if __name__ == "__main__":
    main()