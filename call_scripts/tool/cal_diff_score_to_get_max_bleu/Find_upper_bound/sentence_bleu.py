from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
import os
import sys
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt





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
parser.add_argument('--output-dir', metavar='FILE', type=str,
                    help='output_dir')
parser.add_argument('--output-max-sentence-prob', metavar='FILE', type=str,
                    help='output max_sentence files')
parser.add_argument('--output-index-prob', metavar='FILE', type=str,
                    help='output index files')

parser.add_argument('--source-token-num', metavar='FILE', type=str,
                    help='source_token_num')
parser.add_argument('--target-token-num', metavar='FILE', type=str,
                    help='target_token_num')



# Parse the arguments
args = parser.parse_args()


lines_data=[]
for id , path in enumerate(args.hypo_path) :
    with open(str(path), encoding="utf-8") as f:
        lines_data.append(f.readlines())
print("===========List of experiment=======")
# Extract the values
rate_list = [os.path.basename(path).split('-')[3][:-4] for path in args.hypo_path]
num_experiment=len(rate_list)
print("Number:{} , the rate_list is {}".format(len(rate_list),rate_list))



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
 
source_token_num=[]
with open(str(args.source_token_num), encoding="utf-8") as f:
    source_token_num.append(f.readlines())
target_token_num=[]
with open(str(args.target_token_num), encoding="utf-8") as f:
    target_token_num.append(f.readlines())        


max_bleu_sentence=[]
max_prob_sentence=[]
max_bleu_array=[]
ref_data_flat=[]
max_bleu_index_array=[]
max_prob_index_array=[]

bleu_array=torch.empty((len(lines_data),len(lines_data[0])))
for line_id, lines in tqdm(
                enumerate(zip(*lines_data,ref_data)), 
                desc='processing-rate', total=len(lines_data[0])):
    
    hypo_sents=lines[:-1]
    ref_sent=lines[-1]
    bleu=[]
    sentence=[]
    prob=[]
    for data_id ,hypo_sent in enumerate(hypo_sents):   
        reference=[ref_sent.rstrip("\n").split()]
        candidate=hypo_sent.rstrip("\n").split()[1:]
        
        prob.append(float(hypo_sent.strip("\n").split("\t")[0]))
        sentence.append(hypo_sent.strip("\n").split("\t")[1:])
        smoothie = SmoothingFunction().method1
        bleu.append(sentence_bleu(reference, candidate, smoothing_function=smoothie)*100)
    ref_data_flat.append(reference)
    
    
    max_bleu = max(bleu)
    max_bleu_index = bleu.index(max_bleu)
    max_bleu_array.append(max_bleu) 
    max_bleu_sentence.append(hypo_sents[max_bleu_index].rstrip("\n").split()[1:])
    max_bleu_index_array.append(max_bleu_index)


    max_prob = max(prob)
    max_prob_index = prob.index(max_prob)
    
    max_prob_sentence.append(hypo_sents[max_prob_index].rstrip("\n").split()[1:])
    max_prob_index_array.append(max_prob_index)

print('Max Prob:   {}'.format(corpus_bleu(ref_data_flat,max_prob_sentence,weights=(0.25, 0.25, 0.25, 0.25))*100))
print('Upper Bound:{}'.format(corpus_bleu(ref_data_flat,max_bleu_sentence,weights=(0.25, 0.25, 0.25, 0.25))*100))

save_data(max_bleu_index_array, args.output_index_path)
save_data(max_bleu_array, args.output_bleu_path)




rate_counts_bleu=[]
rate_counts_prob=[]
for i in range(num_experiment) :
    rate_counts_bleu.append(max_bleu_index_array.count(i))
    rate_counts_prob.append(max_prob_index_array.count(i))



# Print index counts
index_count_str_bleu = "Index count - Upper Bound: "
index_count_str_prob = "Index count - Max Prob   : "
labels=[]
for i, count in enumerate(zip(rate_counts_bleu, rate_counts_prob)):
    index_count_str_bleu += "Rate{}: {} / ".format( rate_list[i], count[0])
    index_count_str_prob += "Rate{}: {} / ".format( rate_list[i], count[1])
    labels.append("Rate{}".format(rate_list[i]))    
index_count_str_bleu = index_count_str_bleu.rstrip(" / ")  # Remove trailing " / "
index_count_str_prob = index_count_str_prob.rstrip(" / ")  # Remove trailing " / "
print(index_count_str_prob)
print(index_count_str_bleu)

    
width = 0.35
indices = np.arange(num_experiment)  # 生成一个等差数组作为每个直方图的位置

# fig, ax = plt.subplots(figsize=(8, 6))  # 调整图形大小

fig, ax1 = plt.subplots(1, 1, figsize=(8, 12))  # 創建兩個子圖
# 繪製第一個圖
ax1.bar(indices, rate_counts_bleu, width, label='Bleu')
ax1.bar(indices - width, rate_counts_prob, width, label='Probability')
ax1.set_xlabel('Rates')
ax1.set_ylabel('Count')
ax1.set_title('Count of Different Rates')
ax1.legend()
ax1.set_xticks(indices)
ax1.set_xticklabels(labels)
plt.savefig(args.output_index_fig_path)

# ax.bar(indices - width/2, rate_counts_bleu, width, label='Bleu')
# ax.bar(indices + width/2, rate_counts_prob, width, label='Probability')
# ax.set_xlabel('Rates')
# ax.set_ylabel('Count')
# ax.set_title('Count of Different Rates')
# ax.legend()
# ax.set_xticks(indices)
# ax.set_xticklabels(labels)
# plt.savefig(args.output_index_fig_path)



# import random

# # 將每個字符串元素去除`\n`
# source_token_num = [num.strip() for num in source_token_num[0]]

# # 將列表轉換為整數型別
# source_token_num = [int(num) for num in source_token_num]

# # 將 max_prob_index_array 和 source_token_num 組合成一個二維數組
# data = np.array(list(zip(max_prob_index_array, source_token_num)))

# # 取得不同類別的唯一值
# classes = np.unique(max_prob_index_array)

# # 計算子圖的行數和列數
# num_rows = len(classes)
# num_cols = 1

# # 創建子圖
# fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 6*num_rows))

# # 迭代繪製每個類別的分布圖
# for i, cls in enumerate(classes):
#     # 選擇該類別的數據
#     cls_data = data[data[:, 0] == cls][:, 1]

#     # 繪製直方圖
#     ax = axes[i] if num_rows > 1 else axes  # 如果只有一行子圖，直接使用axes，否則使用axes[i]
#     ax.hist(cls_data, bins='auto')

#     # 設定標籤和標題
#     ax.set_xlabel('Source Token Num')
#     ax.set_ylabel('Count')
#     ax.set_title(f'Distribution of Source Token Num for Class {cls}')

# # 調整子圖間的間距
# plt.tight_layout()

# # 保存圖片
# plt.savefig(args.output_dir + "ggg.png")
# plt.close()  # 關閉圖形




# # 將每個字符串元素去除`\n`
# source_token_num = [num.strip() for num in source_token_num[0]]

# # 將列表轉換為整數型別
# source_token_num = [int(num) for num in source_token_num]

# # 將 max_prob_index_array 和 source_token_num 組合成一個二維數組
# data = np.array(list(zip(max_prob_index_array, source_token_num)))
# # 取得不同類別的唯一值
# classes = np.unique(max_prob_index_array)

# # 計算所有子圖的最大計數值
# max_counts = 0
# for cls in classes:
#     # 選擇該類別的數據
#     cls_data = data[data[:, 0] == cls][:, 1]
#     counts, _ = np.histogram(cls_data, bins='auto')
#     max_counts = max(max_counts, np.max(counts))

# # 創建子圖
# fig, axes = plt.subplots(len(classes), 1, figsize=(8, 6*len(classes)))

# # 迭代繪製每個類別的分布圖
# for i, cls in enumerate(classes):
#     # 選擇該類別的數據
#     cls_data = data[data[:, 0] == cls][:, 1]

#     # 繪製直方圖
#     ax = axes[i] if len(classes) > 1 else axes  # 如果只有一個子圖，直接使用axes，否則使用axes[i]
#     ax.hist(cls_data, bins='auto')

#     # 設定標籤和標題
#     ax.set_xlabel('Source Token Num')
#     ax.set_ylabel('Count')
#     ax.set_title(f'Distribution of Source Token Num for Class {cls}')
    
#     # 設定 x 軸範圍
#     ax.set_xlim([min(source_token_num), max(source_token_num)])
    
#     # 設定 y 軸範圍
#     ax.set_ylim([0, max_counts])

# # 調整子圖間的間距
# plt.tight_layout()

# # 保存圖片
# plt.savefig(args.output_dir + "ggg.png")
# plt.close()  # 關閉圖形









# 將每個字符串元素去除`\n`
source_token_num = [num.strip() for num in source_token_num[0]]

# 將列表轉換為整數型別
source_token_num = [int(num) for num in source_token_num]

# 將 max_prob_index_array 和 source_token_num 組合成一個二維數組
data = np.array(list(zip(max_prob_index_array, source_token_num)))

# 取得不同類別的唯一值
classes = np.unique(max_prob_index_array)

# 創建一個子圖
fig, ax = plt.subplots(figsize=(8, 6))


unique_values, counts = np.unique(source_token_num, return_counts=True)
ax.plot(unique_values, counts, label=f'all')
# 迭代繪製每個類別的折線圖
for cls in classes:
    # 選擇該類別的數據
    cls_data = data[data[:, 0] == cls][:, 1]

    # 計算每個數字的頻率
    unique_values, counts = np.unique(cls_data, return_counts=True)

    # 繪製折線圖
    ax.plot(unique_values, counts, label=f'Class {cls} Trend')

# 設定標籤和標題
ax.set_xlabel('Source Token Num')
ax.set_ylabel('Count')
ax.set_title('Distribution of Max Prob')
ax.legend()

# 保存圖片
plt.savefig(args.output_dir + "Max_Bleu.png")
plt.close()  # 關閉圖形






# 將 max_prob_index_array 和 source_token_num 組合成一個二維數組
data = np.array(list(zip(max_bleu_index_array, source_token_num)))

# 取得不同類別的唯一值
classes = np.unique(max_bleu_index_array)

# 創建一個子圖
fig, ax = plt.subplots(figsize=(8, 6))


unique_values, counts = np.unique(source_token_num, return_counts=True)
ax.plot(unique_values, counts, label=f'all')
# 迭代繪製每個類別的折線圖
for cls in classes:
    # 選擇該類別的數據
    cls_data = data[data[:, 0] == cls][:, 1]

    # 計算每個數字的頻率
    unique_values, counts = np.unique(cls_data, return_counts=True)

    # 繪製折線圖
    ax.plot(unique_values, counts, label=f'Class {cls} Trend')

# 設定標籤和標題
ax.set_xlabel('Source Token Num')
ax.set_ylabel('Count')
ax.set_title('Distribution of Max Prob')
ax.legend()

# 保存圖片
plt.savefig(args.output_dir + "Max_Prob.png")
plt.close()  # 關閉圖形











# # 將每個字符串元素去除`\n`
# source_token_num = [num.strip() for num in source_token_num[0]]

# # 將列表轉換為整數型別
# source_token_num = [int(num) for num in source_token_num]

# # # 將 max_prob_index_array 和 source_token_num 組合成一個二維數組
# # data = np.array(list(zip(max_prob_index_array, source_token_num)))

# # # 取得不同類別的唯一值
# # classes = np.unique(max_prob_index_array)

# # for cls in classes:
# #     # 選擇該類別的數據
# #     cls_data = data[data[:, 0] == cls][:, 1]
    
# #     # 繪製該類別的直方圖
# #     ax2.hist(cls_data, bins='auto', label=f'Class {cls}')
    
# # # 設定標籤和標題
# # ax2.set_xlabel('Source Token Num')
# # ax2.set_ylabel('Count')
# # ax2.set_title('Distribution of Source Token Num by Max Prob Index')
# # ax2.legend()
# # # ax2.set_xticks(indices)
# # plt.savefig(args.output_dir + "ggg.png")


# # 將 max_prob_index_array 和 source_token_num 組合成一個二維數組
# data = np.array(list(zip(max_prob_index_array, source_token_num)))

# # 取得不同類別的唯一值
# classes = np.unique(max_prob_index_array)

# # 迭代繪製每個類別的分布圖
# for cls in classes:
#     # 選擇該類別的數據
#     cls_data = data[data[:, 0] == cls][:, 1]

#     # 創建一個子圖
#     fig, ax = plt.subplots(figsize=(8, 6))

#     # 計算每個數據點的出現次數
#     unique_values, counts = np.unique(cls_data, return_counts=True)

#     # 繪製散點圖
#     ax.scatter(unique_values, counts)

#     # 設定標籤和標題
#     ax.set_xlabel('Source Token Num')
#     ax.set_ylabel('Count')
#     ax.set_title(f'Distribution of Source Token Num for Class {cls}')
    
#     # 保存圖片
#     plt.savefig(args.output_dir + f"Class_{cls}_ggg.png")
#     plt.close()  # 關閉圖形

