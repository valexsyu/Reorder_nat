import argparse
from tqdm import tqdm
import pdb

from matplotlib import pyplot as plt
import numpy as np

def add_args(parser):
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="data path",
    ) 
    parser.add_argument(
        "--data-token-dir",
        type=str,
        default=None,
        help="token data input path",
    )     
    parser.add_argument(
        "--output-token-dir",
        type=str,
        default=None,
        help="token data output path",
    )         
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="output data path",
    )     
    parser.add_argument(
        "--src",
        type=str,
        default="de",
        help="source language",
    )
    parser.add_argument(
        "--tgt",
        type=str,
        default="en",
        help="target language",
    )   

def read_data(data_type, root_path, data_token_path,src_lang, tgt_lang):
    align_data_path = root_path + data_type + ".align.de-en"
    with open(align_data_path, encoding="utf-8") as f:
        align_lines = f.readlines()
    
    src_data_path = root_path + data_type + "." + src_lang
    with open(src_data_path, encoding="utf-8") as f:
        src_lines = f.readlines()

    tgt_data_path = root_path + data_type + "." + tgt_lang
    with open(tgt_data_path, encoding="utf-8") as f:
        tgt_lines = f.readlines()
   
    src_data_token_path = data_token_path + data_type + "." + src_lang
    with open(src_data_token_path, encoding="utf-8") as f:
        src_token_lines = f.readlines()

    
    return align_lines, src_lines, tgt_lines, src_token_lines

def plothist(src_num, src_complete_num,data_type,save_path):
    plt.hist(src_num)
    plt.savefig(save_path + str(data_type) +".png")
    plt.hist(src_complete_num)
    plt.savefig(save_path + str(data_type) +".png")  
    plt.clf()  

def save_data(src_align, data_type, save_path) :
# open file in write mode
    with open(save_path + str(data_type) +  '.align' , 'w') as fp:
        fp.writelines(' '.join(str(j) for j in i) + '\n' for i in src_align)
            

    

def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    src_lang = args.src     
    tgt_lang = args.tgt
    data_path = args.data_dir
    data_token_path = args.data_token_dir
    data_types = ["test" , "valid", "train"]
    output_path = args.output_dir
    output_token_path = args.output_token_dir
    for data_type in data_types:
        align_lines, src_lines, tgt_lines, src_token_lines = \
                    read_data(str(data_type),data_path, data_token_path, src_lang, tgt_lang)
        src_align = []
        src_len = []
        src_complete_len = []
        for line_id, (align_line, src_line, tgt_line) in tqdm(
                        enumerate(zip(align_lines, src_lines, tgt_lines)), 
                        desc='processing', total=len(align_lines)):
            src_line = src_line.strip("\n").split(" ")
            #tgt_line = tgt_line.strip("\n").split(" ")
            align_line = align_line.strip("\n").split(" ")
            len_src = len(src_line)
            src_index = 0
            #len_tgt = len(tgt_line)    
            src_align_line = []
            src_align_pos_line = []
            pre_tgt_pos = 0
            for align_idx, align in enumerate(align_line) :
                src_align_pos , tgt_align_pos = align.split("-")
                src_align_pos , tgt_align_pos = int(src_align_pos) , int(tgt_align_pos)
                if src_align_pos > src_index :
                    for i in range(src_index, src_align_pos):
                        src_align_line.extend([str(pre_tgt_pos)])
                    src_align_line.extend([str(tgt_align_pos)])
                else :
                    src_align_line.extend([str(tgt_align_pos)])
                pre_tgt_pos = tgt_align_pos
                src_index = src_align_pos + 1
                src_align_pos_line.extend([str(src_align_pos)])
            src_set_line = set(src_align_pos_line)
            if len(src_set_line) ==  len_src :
                src_complete_len.append(len_src)
            src_align.append(src_align_line)
            src_len.append(len_src)
        plothist(src_len, src_complete_len, data_type, output_path)
        save_data(src_align, data_type, output_path)

        ##token
        src_token_align = []
        for line_id , (src_token_line, align_line) in enumerate(zip(src_token_lines, src_align)):
            src_token_line = src_token_line.strip("\n").split(" ")
            src_token_align_line = []
            pos = 0
            cur_align_value = align_line[pos]
            for token_idx, token in enumerate(src_token_line) :
                if token[:2] == "##" :
                    src_token_align_line.extend([str(token_idx)+'-'+str(pre_align_value)])
                else :
                    src_token_align_line.extend([str(token_idx)+'-'+str(cur_align_value)])
                    pos += 1
                    pre_align_value = cur_align_value
                    if pos <= len(align_line)-1 :
                        cur_align_value = align_line[pos]

            
            src_token_align.append(src_token_align_line)
        save_data(src_token_align, data_type, output_token_path)
                
                
                    




        print(str(data_type) + ' Done')


            


if __name__ == "__main__":
    main()
