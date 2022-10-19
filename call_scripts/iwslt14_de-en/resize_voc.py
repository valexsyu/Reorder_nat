from tqdm import tqdm
import os
def read_data(data_file):
    with open(data_file, encoding="utf-8") as f:
        src_lines = f.readlines()
    return src_lines


def main():
    root_path="/home/valexsyu/Doc/NMT/Reorder_nat/data/mbert_extract_voc/iwslt14_de_en"
    voc_file = "tgt_vocab.txt"
    data_sets = ["train.de" ,"train.en" ,"valid.de" ,"valid.en" ,"distilled_train.en", "distilled_valid.en"]
    voc_file = os.path.join(root_path , voc_file)
    
    with open(voc_file) as f:
        remove_voc = f.read().splitlines()    
    original_voc = remove_voc.copy()
    for data_set in data_sets :
        data_file = os.path.join(root_path , data_set)
        with open(data_file) as f:
            data_lines = f.read().splitlines()            
        for line_id, line in tqdm(enumerate(data_lines), desc=data_set, total=len(data_lines)):
            line = line.strip("\n").split(" ")
            for token in line :
                if token in remove_voc :
                    remove_voc.remove(token)
    wanted_voc = list(set(original_voc) - set(remove_voc))
    wanted_file = os.path.join(root_path , "wanted_voc.txt")
    with open(wanted_file, 'w') as f:
        for line in wanted_voc:
            f.write(f"{line}\n")
    
    
      
if __name__ == "__main__":
    main()