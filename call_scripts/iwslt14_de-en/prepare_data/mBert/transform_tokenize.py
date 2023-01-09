from transformers import AutoTokenizer, AutoModel, BertTokenizer
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='input file')
    parser.add_argument('--output', type=str, required=True, help='output file')
    parser.add_argument('--pretrained_model', type=str, required=True, help='pretrained language model')  
    args = parser.parse_args()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.pretrained_model)
    fo = open(args.input, encoding="utf-8")
    fw = open(args.output, "w", encoding="utf-8")
    lines =  [line[:-1].strip() for line in fo.readlines()]
    tokenized_ids = tokenizer.batch_encode_plus(lines, add_special_tokens=False)['input_ids']
    for i in tqdm(range(len(tokenized_ids))):
        res = ' '.join(tokenizer.convert_ids_to_tokens(tokenized_ids[i]))
        # if '[UNK]' in res:
        #     import pdb; pdb.set_trace()
        fw.write(res + '\n')      

    fo.close()
    fw.close()

if __name__ == "__main__":
  main()
  
  
