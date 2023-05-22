from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os
import sys
from tqdm import tqdm

ref_path = sys.argv[1]
hypo_path = sys.argv[2]
output_path = sys.argv[3]




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
parser.add_argument('--hypo_path', metavar='FILE', type=str, nargs='+',
                    help='input data files')
parser.add_argument('--ref_path', metavar='FILE', type=str, nargs='+',
                    help='input target files')

parser.add_argument('--out_path', metavar='FILE', type=str,
                    help='output max_sentence files')
parser.add_argument('--output_index', metavar='FILE', type=str,
                    help='output index files')

# Parse the arguments
args = parser.parse_args()


import pdb;pdb.set_trace()
lines_data=[]
for id , path in enumerate(args.hypo_path) :
    with open(str(path), encoding="utf-8") as f:
        lines_data.append(f.readlines())

ref_data=[]
for id , path in enumerate(args.ref_path) :
    with open(str(path), encoding="utf-8") as f:
        ref_data.append(f.readlines())


def save_data(data_in,file_name) :
    # open file in write mode
    with open( file_name, 'w') as fp:
        fp.writelines('\n'.join(data_in))  



max_sentence=[]
max_index_array=[]
for line_id, lines in tqdm(
                enumerate(zip(*lines_data)), 
                desc='processing-rate', total=len(lines_data[0])):
    import pdb;pdb.set_trace()
    
    bleu_array=[]
    for data_id ,data_line in tqdm(enumerate(zip(*lines,ref_data)),
                              desc='processing-line', total=len(lines)):
        
        reference=[lines[0].rstrip("\n").split()]
        candidate=lines[1].rstrip("\n").split()
        smoothie = SmoothingFunction().method1
        bleu = sentence_bleu(reference, candidate, smoothing_function=smoothie)*100
        bleu_array.append(bleu)

        
    max_value = max(value)
    max_index = value.index(max_value)
    
    max_sentence.append(sentence[max_index])
    max_index_array.append(str(max_index))    

save_data(max_sentence, args.output_max_sentence)
save_data(max_index_array, args.output_index)





ref_data=[]
with open(str(ref_path), encoding="utf-8") as f:
    ref_data.append(f.readlines())

hoyp_data=[]
with open(str(hypo_path), encoding="utf-8") as f:
    hoyp_data.append(f.readlines())

def save_data(data_in, file_name):
    # Convert float values to strings
    data_strings = [str(item) for item in data_in]

    # Open file in write mode
    with open(file_name, 'w') as fp:
        fp.writelines('\n'.join(data_strings))


bleu_array=[]
for line_id, lines in tqdm(
                enumerate(zip(ref_data[0],hoyp_data[0])), 
                desc='sentence-bleu-procesing', total=len(ref_data[0])):
    reference=[lines[0].rstrip("\n").split()]
    candidate=lines[1].rstrip("\n").split()
    
    smoothie = SmoothingFunction().method1
    bleu = sentence_bleu(reference, candidate, smoothing_function=smoothie)*100
    bleu_array.append(bleu)
    
save_data(bleu_array, output_path)   
