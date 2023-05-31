

# example python call_scripts/tool/load_checkpoint_step.py checkpoints/m-B-3-1-N-UR25M both
import torch
import os
import sys
from tqdm import tqdm

def load_last_optimizer_state(load_file_path):
    return torch.load(load_file_path, map_location=torch.device('cpu'))['last_optimizer_state']

def load_file(last_optimizer_state, output_list, file_class=""):
    if 'param_groups' in last_optimizer_state and 'step' in last_optimizer_state['param_groups'][0]:
        output_list.append(file_class + str(last_optimizer_state['param_groups'][0]['step']))
    elif 'state' in last_optimizer_state and 'step' in last_optimizer_state['state'][0]:
        output_list.append(file_class + str(last_optimizer_state['state'][0]['step']))
    else:
        optimizer_group=list(last_optimizer_state.keys())
        output_list.append(file_class + str(last_optimizer_state[optimizer_group[0]]['param_groups'][0]['step'])) 
    
    return output_list   
         

def best_file_class(input_path,output_list):
    file_class=""
    file_names = [name for name in os.listdir(input_path) if name.startswith('checkpoint.best_bleu')]
    # for name in file_names:
    for name in tqdm(file_names, disable=True, desc='Loading best files'):
        if name.startswith('checkpoint.best_bleu'):
            i=+1
            load_file_path=os.path.join(input_path,name)
            last_optimizer_state=load_last_optimizer_state(load_file_path)
            output_list = load_file(last_optimizer_state, output_list, file_class)
    return output_list

def last_file_class(input_path,output_list):   
    file_class="last:"       
    load_file_path=os.path.join(input_path,'checkpoint_last.pt')
    last_optimizer_state=load_last_optimizer_state(load_file_path)
    output_list = load_file(last_optimizer_state, output_list, file_class)
    
    return output_list         


input_path = sys.argv[1]
additional_arg = "both"

if len(sys.argv) > 2:
    additional_arg = sys.argv[2]
    

output_list=['model_step:']    
if additional_arg == "both" :
    output_list = best_file_class(input_path,output_list)
    output_list = last_file_class(input_path,output_list)
elif additional_arg == "best" :
    output_list = best_file_class(input_path,output_list)
elif additional_arg == "last" :
    output_list = last_file_class(input_path,output_list)
else:
    print("==================Error========================")
    print(additional_arg)
    print("Error ~~")
    import pdb;pdb.set_trace()
    print("Error ~~")
    
print(output_list)