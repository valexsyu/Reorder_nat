import torch, os , sys

def load_last_optimizer_state(file_path):
    return torch.load(load_file_path)['last_optimizer_state']
    

input_path = sys.argv[1]
output_list=['model_step:']
for name in os.listdir(input_path):
    if name.startswith('checkpoint.best_bleu'):
        i=+1
        load_file_path=os.path.join(input_path,name)
        last_optimizer_state=load_last_optimizer_state(load_file_path)
        if 'param_groups' in last_optimizer_state and 'step' in last_optimizer_state['param_groups'][0]:
            output_list.append(last_optimizer_state['param_groups'][0]['step'])
        elif 'state' in last_optimizer_state and 'step' in last_optimizer_state['state'][0]:
            output_list.append(last_optimizer_state['state'][0]['step'])
        else:
            optimizer_group = list(last_optimizer_state.keys())
            output_list.append(last_optimizer_state[optimizer_group[0]]['param_groups'][0]['step'])
            output_list[0]=optimizer_group[0]+'_step'
        
        
load_file_path=os.path.join(input_path,'checkpoint_last.pt')
last_optimizer_state=load_last_optimizer_state(load_file_path)
if 'param_groups' in last_optimizer_state and 'step' in last_optimizer_state['param_groups'][0]:
    output_list.append('last:' + str(last_optimizer_state['param_groups'][0]['step']))
elif 'state' in last_optimizer_state and 'step' in last_optimizer_state['state'][0]:
    output_list.append('last:' + str(last_optimizer_state['state'][0]['step']))
else:
    optimizer_group=list(last_optimizer_state.keys())
    output_list.append('last:' + str(last_optimizer_state[optimizer_group[0]]['param_groups'][0]['step']))           

print(output_list)

# input_path = sys.argv[1]
# output_list=[]
# for name in os.listdir(input_path):
#     if name.startswith('checkpoint.best_bleu'):
#         i=+1
#         load_file_path=os.path.join(input_path,name)
#         import pdb;pdb.set_trace()
#         if torch.load(load_file_path)['last_optimizer_state']['param_groups'][0].get('step',None) is not None:
#             output_list.append(torch.load(load_file_path)['last_optimizer_state']['param_groups'][0]['step'])
#         else:
#             output_list.append(torch.load(load_file_path)['last_optimizer_state']['state'][0]['step'])
        
        
# load_file_path=os.path.join(input_path,'checkpoint_last.pt')
# if torch.load(load_file_path)['last_optimizer_state']['param_groups'][0].get('step',None) is not None:
#     output_list.append('last:' + str(torch.load(load_file_path)['last_optimizer_state']['param_groups'][0]['step']))  
# else:
#     output_list.append('last:' + str(torch.load(load_file_path)['last_optimizer_state']['state'][0]['step']))              
# print(output_list)