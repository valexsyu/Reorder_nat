from tqdm import tqdm
import pdb


import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description='Process some files.')

# Add the file arguments
parser.add_argument('--data', metavar='FILE', type=str, nargs='+',
                    help='input data files')
parser.add_argument('--tgt', metavar='FILE', type=str, nargs='+',
                    help='input target files')

parser.add_argument('--output_max_sentence', metavar='FILE', type=str,
                    help='output max_sentence files')
parser.add_argument('--output_index', metavar='FILE', type=str,
                    help='output index files')

# Parse the arguments
args = parser.parse_args()

# Print the input files
# print("Data files:", args.data)
# print("Target files:", args.tgt)

lines_data=[]
f=[]
for id , data in enumerate(args.data) :
    with open(str(data), encoding="utf-8") as f:
        lines_data.append(f.readlines())


def save_data(data_in,file_name) :
    # open file in write mode
    with open( file_name, 'w') as fp:
        fp.writelines('\n'.join(data_in))  



max_sentence=[]
max_index_array=[]
for line_id, lines in tqdm(
                enumerate(zip(*lines_data)), 
                desc='processing-1', total=len(lines_data[0])):
    sentence=[]
    value=[]
    for data_id ,data_line in enumerate(lines):
        value.append(float(data_line.strip("\n").split("\t")[0]))
        sentence.append(data_line.strip("\n").split("\t")[1])
        
    max_value = max(value)
    max_index = value.index(max_value)
    
    max_sentence.append(sentence[max_index])
    max_index_array.append(str(max_index))    

save_data(max_sentence, args.output_max_sentence)
save_data(max_index_array, args.output_index)

       

    

    
        