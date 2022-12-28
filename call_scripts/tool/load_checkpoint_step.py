import torch, os , sys

input_path = sys.argv[1]
output_list=[]
for name in os.listdir(input_path):
    if name.startswith('checkpoint.best_bleu'):
        i=+1
        load_file_path=os.path.join(input_path,name)
        if torch.load(load_file_path)['last_optimizer_state']['param_groups'][0]['step'] is not None:
            output_list.append(torch.load(load_file_path)['last_optimizer_state']['param_groups'][0]['step'])
        else:
            output_list.append(torch.load(load_file_path)['last_optimizer_state']['state'][0]['step'])
            
print(output_list)