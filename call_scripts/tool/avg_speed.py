import argparse
from tqdm import tqdm
import pdb

from matplotlib import pyplot as plt
import numpy as np
from multiprocessing import Pool


def add_args(parser):
    parser.add_argument('input_values', metavar='N', type=float, nargs='+',
                    help='an integer for the accumulator')

def Average(lst):
    return sum(lst) / len(lst)

def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    # import pdb;pdb.set_trace()
    input_list = args.input_values

    
    # Driver Code
    average = Average(input_list)
    
    # Printing average of the list
    print("", round(average, 3))
        

            


if __name__ == "__main__":
    main()