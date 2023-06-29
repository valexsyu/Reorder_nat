from tqdm import tqdm
import pdb
import numpy as np
import matplotlib.pyplot as plt

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

parser.add_argument('--source_token_num', metavar='FILE', type=str,
                    help='source_token_num')
parser.add_argument('--target_token_num', metavar='FILE', type=str,
                    help='target_token_num')
parser.add_argument('--output_dir', metavar='FILE', type=str,
                    help='output_dir')

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

source_token_num=[]
with open(str(args.source_token_num), encoding="utf-8") as f:
    source_token_num.append(f.readlines())
target_token_num=[]
with open(str(args.target_token_num), encoding="utf-8") as f:
    target_token_num.append(f.readlines())
    
rate_number=len(args.data)
print("rate_number:{}".format(rate_number))
    




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

       

class_counts = np.bincount(max_index_array, minlength=rate_number)

# 统计每个类别的数量
class_counts = np.bincount(max_index_array, minlength=rate_number)

# 生成横坐标
# x = np.arange(rate_number)

# 绘制柱状图
# plt.bar(x, class_counts)

# 添加标签和标题
# plt.xlabel('Class')
# plt.ylabel('Count')
# plt.title('Count of Different Classes')
print(args.output_dir+"ggg.png")
# plt.savefig(args.output_dir+"/ggg.png")        


# Print index counts
index_count_str = "Index count: "
labels=[]
for i, count in enumerate(class_counts):
    index_count_str += "Rate{}: {} / ".format( values[i], count)
    labels.append("Rate{}".format(values[i]))
index_count_str = index_count_str.rstrip(" / ")  # Remove trailing " / "
print(index_count_str)

    
# # Define the labels for the types

# Plot the distribution
plt.bar(labels, class_counts)
plt.xlabel('Rates')
plt.ylabel('Count')
plt.title('Distribution of Rates')
plt.savefig(args.output_dir+"/ggg.png")  