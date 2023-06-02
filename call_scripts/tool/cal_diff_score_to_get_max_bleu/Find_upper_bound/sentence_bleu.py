from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
import os
import sys
from tqdm import tqdm
import numpy as np
import torch

# ref_path = sys.argv[1]
# hypo_path = sys.argv[2]
# output_path = sys.argv[3]




# ref_data=[]
# with open(str(ref_path), encoding="utf-8") as f:
#     ref_data.append(f.readlines())

# hoyp_data=[]
# with open(str(hypo_path), encoding="utf-8") as f:
#     hoyp_data.append(f.readlines())

# def save_data(data_in, file_name):
#     # Convert float values to strings
#     data_strings = [str(item) for item in data_in]

#     # Open file in write mode
#     with open(file_name, 'w') as fp:
#         fp.writelines('\n'.join(data_strings))


# bleu_array=[]
# for line_id, lines in tqdm(
#                 enumerate(zip(ref_data[0],hoyp_data[0])), 
#                 desc='sentence-bleu-procesing', total=len(ref_data[0])):
#     reference=[lines[0].rstrip("\n").split()]
#     candidate=lines[1].rstrip("\n").split()
    
#     smoothie = SmoothingFunction().method1
#     bleu = sentence_bleu(reference, candidate, smoothing_function=smoothie)*100
#     bleu_array.append(bleu)
    
# save_data(bleu_array, output_path)    



import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description='Process some files.')

# Add the file arguments
parser.add_argument('--hypo-path', metavar='FILE', type=str, nargs='+',
                    help='input data files')
parser.add_argument('--ref-path', metavar='FILE', type=str, nargs='+',
                    help='input target files')

parser.add_argument('--output-bleu-path', metavar='FILE', type=str,
                    help='output max_sentence files')
parser.add_argument('--output-index-path', metavar='FILE', type=str,
                    help='output index files')
parser.add_argument('--output-bleu-fig-path', metavar='FILE', type=str,
                    help='output index files')
parser.add_argument('--output-index-fig-path', metavar='FILE', type=str,
                    help='output index files')


# Parse the arguments
args = parser.parse_args()


lines_data=[]
for id , path in enumerate(args.hypo_path) :
    with open(str(path), encoding="utf-8") as f:
        lines_data.append(f.readlines())

ref_data=[]
with open(args.ref_path[0], encoding="utf-8") as f:
    ref_data.extend(f.readlines())


def save_data(data_in,file_name) :
    # open file in write mode
    with open( file_name, 'w') as fp:
        fp.writelines('\n'.join(data_in))  


def save_data(data_in, file_name):
    # Convert float values to strings
    data_strings = [str(item) for item in data_in]

    # Open file in write mode
    with open(file_name, 'w') as fp:
        fp.writelines('\n'.join(data_strings))


max_sentence=[]
max_bleu_array=[]
ref_data_flat=[]
max_index_array=[]
bleu_array=torch.empty((len(lines_data),len(lines_data[0])))
for line_id, lines in tqdm(
                enumerate(zip(*lines_data,ref_data)), 
                desc='processing-rate', total=len(lines_data[0])):
    
    hypo_sents=lines[:-1]
    ref_sent=lines[-1]
    bleu=[]
    for data_id ,hypo_sent in enumerate(hypo_sents):   
        reference=[ref_sent.rstrip("\n").split()]
        candidate=hypo_sent.rstrip("\n").split()
        smoothie = SmoothingFunction().method1
        # bleu_array[data_id][line_id] = sentence_bleu(reference, candidate, smoothing_function=smoothie)*100
        bleu.append(sentence_bleu(reference, candidate, smoothing_function=smoothie)*100)
    ref_data_flat.append(reference)
    
    
    max_bleu = max(bleu)
    max_index = bleu.index(max_bleu)
       
    max_bleu_array.append(max_bleu) 
    max_sentence.append(hypo_sents[max_index].rstrip("\n").split())
    max_index_array.append(max_index)


print('Upper Bound:{}'.format(corpus_bleu(ref_data_flat,max_sentence,weights=(0.25, 0.25, 0.25, 0.25))*100))

save_data(max_index_array, args.output_index_path)
save_data(max_bleu_array, args.output_bleu_path)



import matplotlib.pyplot as plt


# Count the occurrences of each type
rate2_num, rate3_num, rate4_num = max_index_array.count(0), max_index_array.count(1), max_index_array.count(2)
counts = [rate2_num, rate3_num, rate4_num ]

# Define the labels for the types
labels = ['Rate 2', 'Rate 3', 'Rate 4']

# Plot the distribution
plt.bar(labels, counts)
plt.xlabel('Rates')
plt.ylabel('Count')
plt.title('Distribution of Rates')
plt.savefig(args.output_index_fig_path)

print("Index count: Rate2: {} / Rate3: {} / Rate4:{}".format(rate2_num, rate3_num, rate4_num ))


