import argparse
from tqdm import tqdm
import pdb

from matplotlib import pyplot as plt
import numpy as np
from multiprocessing import Pool


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
def read_data(data_type, root_path):
    data_path = root_path + "num_" + data_type + ".txt"
    with open(data_path, encoding="utf-8") as f:
        lines = f.readlines()
    return lines

def plothist(src_num, src_complete_num,data_type,save_path):
    plt.hist(src_num)
    plt.savefig(save_path + str(data_type) +".png")
    plt.hist(src_complete_num)
    plt.savefig(save_path + str(data_type) +".png")  
    plt.clf()  
def plot2d(data_input, data_type, save_path, post_fix):
    plt.plot(data_input[0])
    plt.plot(data_input[1])
    plt.savefig(save_path + str(data_type) + post_fix +".png")
    plt.clf()      

def save_data(src_align, data_type, save_path) :
# open file in write mode
    with open(save_path + str(data_type) +  '.align' , 'w') as fp:
        fp.writelines(' '.join(str(j) for j in i) + '\n' for i in src_align)
            
def process(data_types,data_path):
    for data_type in data_types:
        lines = read_data(str(data_type), data_path)
        srcs = np.array([])
        tgts = np.array([])
        for line_id, line in tqdm(
                        enumerate(lines), 
                        desc='processing-1', total=len(lines)):
            line = line.strip("\n").split(" ")
            src = line[0]
            tgt = line[1]
            srcs = np.append(srcs,[src])
            tgts = np.append(tgts,[tgt])
        srcs=srcs.astype(np.int)
        tgts=tgts.astype(np.int)
        max_src = np.max(srcs)
        plot2d([srcs, tgts] , data_type, data_path, "-num") 
        plothist(srcs, tgts, data_type, data_path)
        diffs = np.zeros((2, max_src))
        diffs[0][:] = 0
        diffs[1][:] = 0
        for line_id , (src, tgt) in tqdm(enumerate(zip(srcs,tgts)), desc='processing-2', total=len(srcs)):
            src = src - 1
            tgt = tgt - 1
            diff = src - tgt
            if diffs[0][src] > diff :
                diffs[0][src] = diff
            if diffs[1][src] < diff :
                diffs[1][src] = diff

        print(max_src)
        plot2d(diffs , data_type, data_path, "-diff")    

def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    data_types = ["test" , "valid" , "train"]
    data_path = args.data_dir
    process(data_types,data_path)
    # p = Pool(50)
    # p.map(process(data_types,data_path))


        

            


if __name__ == "__main__":
    main()
