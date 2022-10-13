import argparse
from tqdm import tqdm 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default="train.de",  help='path or name of the de data')
    parser.add_argument('--ref-file', type=str, default="train.de",  help='path or name of the reference de data')
    parser.add_argument('--output-file', type=str, default="train.modify.de" ,  help='path or name of de data')
    args = parser.parse_args()  

    with open(args.input_file, "r", encoding="utf-8") as f:
        src_lines = f.readlines()
    with open(args.ref_file, "r", encoding="utf-8") as f:
        ref_lines = f.readlines()        
    with open(args.output_file, "w", buffering=1, encoding="utf-8") as f:
        special_word_modify(src_lines, ref_lines,f )
        

def special_word_modify(src_lines,ref_lines, out_file):
    for line_id, src_line in enumerate(tqdm(src_lines)):
        ref_line = ref_lines[line_id]
        src_line = (src_line.replace('\t'," ").replace("\n","")).strip(" ").split(" ")
        ref_line = (ref_line.replace('\t'," ")).strip(" ").split(" ")
        out_line = src_line
        for _ , ref_word in enumerate((ref_line)):
            for i, sp_letter in enumerate(["ä", "ö", "ü"]):
                if sp_letter in ref_word:
                    if sp_letter == "ä" :
                        temp_word = ref_word.replace("ä","a")
                    elif sp_letter == "ö" :
                        temp_word = ref_word.replace("ö","o") 
                    else:
                        temp_word = ref_word.replace("ü","u")
                    out_line = [ref_word if x==temp_word else x for x in out_line]
        print(' '.join(out_line) , file=out_file)
                 
        
if __name__ == "__main__":
    main()