from matplotlib import pyplot as plt 
import argparse
from tqdm import tqdm
import pdb

def read_data(path):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    return lines

def add_args(parser):
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="data path",
    ) 
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="output data path",
    )   
    parser.add_argument(
        "--bibert-path",
        type=str,
        default=None,
        help="output data path",
    )     

def plt_hist(input_path, bibert_path, output_path):
    def pre_process(data_path) :
        lines = read_data(data_path)
        tot_entropys=[]
        for line_id, line in tqdm(enumerate(lines), desc='processing', total=len(lines)):
            line = line.strip("\n").split(" ")
            tot_entropys.extend(line)
        return list(map(float, tot_entropys))
    
    nat = pre_process(input_path)
    bibert = pre_process(bibert_path)
    fig, ax = plt.subplots(1, 1)    
    ax.set_xlabel('Token Entropy')
    ax.set_ylabel('Count')
    _ = ax.hist((nat, bibert), bins=20)
    # Add a legend
    ax.legend(('NAR', 'AR'), loc='upper right')
    fig.savefig(output_path + "entropy" +".png")  

def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    input_path = args.input_path
    bibert_path = args.bibert_path
    output_path = args.output_path
    plt_hist(input_path, bibert_path, output_path)
    print("Save Histogram done")

if __name__ == "__main__":
    main()
