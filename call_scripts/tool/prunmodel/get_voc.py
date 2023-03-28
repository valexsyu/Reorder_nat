from transformers import XLMRobertaTokenizer
import sys

model_path = sys.argv[1]
dest_path = sys.argv[2]
tokenizer = XLMRobertaTokenizer.from_pretrained(f'{model_path}')
vocab = tokenizer.get_vocab()

vocab_map = dict(zip(vocab.values(), vocab.keys()))

with open(f'{dest_path}/vocab.txt', 'w') as f:
    for i in range(len(vocab_map)) :
        f.write(f'{vocab_map[i]}\n')
        
print('Get Vocab done')        
    