from nltk.translate.bleu_score import corpus_bleu
import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description='Process some files.')

# Add the file arguments
parser.add_argument('--hypo-path', metavar='FILE', type=str,
                    help='input data files')
parser.add_argument('--ref-path', metavar='FILE', type=str,
                    help='input target files')



# Parse the arguments
args = parser.parse_args()


with open(str(args.hypo_path), encoding="utf-8") as f:
    hypo_data=f.readlines()


ref_data=[]
with open(args.ref_path, encoding="utf-8") as f:
    ref_data.extend(f.readlines())

ref=[]
hypo=[]
for data_id ,data in enumerate(zip(ref_data,hypo_data)):   
    ref.append([data[0].rstrip("\n").split()])
    hypo.append(data[1].rstrip("\n").split())


print('corpus_bleu:{}'.format(corpus_bleu(ref,hypo,weights=(0.25, 0.25, 0.25, 0.25))*100))
