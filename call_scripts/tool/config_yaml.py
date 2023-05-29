import yaml
import os


from tqdm import tqdm
import pdb


import argparse


def safe_getattr(obj, k, default=None):
    """Returns obj[k] if it exists and is not None, otherwise returns default."""
    from omegaconf import OmegaConf

    if OmegaConf.is_config(obj):
        return obj[k] if k in obj and obj[k] is not None else default

    return getattr(obj, k, default)


# def base_architecture(args):
    # args.output_path = safe_getattr(args, "output_path", "checkpoints/")
    # args.basic_yaml_path = safe_getattr(args, "basic_yaml_path", "call_scripts/train/basic.yaml")    
    # args.output_path = safe_getattr(args, "model_name", "nonautoregressive_reorder_translation")
    # args.criterion_name = safe_getattr(args, "criterion_name", "nat_ctc_loss")   
    # args.max_token = safe_getattr(args, "max_token", 2048) 
    # args.task_name = safe_getattr(args, "task_name", "transaltion_ctcpmlm")       

def parser_function():
    parser = argparse.ArgumentParser(description='Description of your program')
    # parser.add_argument('-f', '--file', help='Input file path', required=True)
    # parser.add_argument('-n', '--number', help='Number of items to process', default=10, type=int)
    # parser.add_argument('-v', '--verbose', help='Enable verbose output', action='store_true')
    # parser.add_argument('-b', '--basic_yaml_path', help='load basic yaml', default="call_scripts/train/basic.yaml", type=str)
    # parser.add_argument('-o', '--output_path', help='save the yaml', default="checkpoints/", type=str)
    # parser.add_argument('-e', '--experiment', help='experiment id', default=None, type=str)
    # parser.add_argument('-m', '--model_name', help='model_name', default='nonautoregressive_reorder_translation', type=str)
    # parser.add_argument('-c', '--criterion_name', help='criterion name', default='nat_ctc_loss', type=str)
    # parser.add_argument('-t', '--task_name', help='task name', default='transaltion_ctcpmlm', type=str)
    # parser.add_argument('--dryrun', help='no reocde in wandb', action='store_true')
    # parser.add_argument('--max_token', help='Number of items to process', default=2048, type=int)
    # parser.add_argument('--fp16',help='use fp16', action='store_true')
    # args = parser.parse_args()
    #checkpoint
    parser.add_argument('-b', '--basic-yaml-path', help='load basic yaml', default="call_scripts/train/basic.yaml", type=str)
    parser.add_argument('-o', '--save-dir', help='save the yaml', default=None, type=str)
    parser.add_argument('-e', '--experiment', help='experiment id', default=None, type=str)
    parser.add_argument('--save-interval-updates', help='save-interval-updates', default=None, type=int)
    
    #common
    parser.add_argument('--wandb-entity', help='no reocde in wandb', default=None, type=str)
    parser.add_argument('--wandb-project', help='no reocde in wandb', default=None, type=str)
    parser.add_argument('--fp16',help='use fp16', action='store_true')
    #criterion
    parser.add_argument('-c', '--criterion-name', help='criterion name', default=None, type=str)
    #dataset
    parser.add_argument('--train-subset', help='criterion name', default=None, type=str)
    parser.add_argument('--max-tokens', help='Number of items to process', default=None, type=int)
    #distributed_training
    parser.add_argument('--distributed-world-size', help='Number of gpu', default=None, type=int)
    #model
    parser.add_argument('-m', '--model-name', help='model_name', default=None, type=str)
    parser.add_argument('--num-upsampling-rate', help='num-upsampling-rate', default=None, type=int)
    parser.add_argument('--embedding-frozen', help='embedding frozen', action='store_true')
    parser.add_argument('--lm-head-frozen', help='lm head frozen', action='store_true')
    parser.add_argument('--upsample-fill-mask', help='upsampling token is used by masking token to fill', action='store_true')
    parser.add_argument('--dynamic-rate', help='use floating rate not integer rate', action='store_true')
    parser.add_argument('--lm-start-step', help='start to add loss by useing lm model (ED loss) ', default=None, type=int)
    parser.add_argument('--voc-choosen', help='others function', default=None, type=int)
    parser.add_argument('--dropout', help='dropout', default=None, type=float)
    parser.add_argument('--has-eos', help='has eos', action='store_true')
    parser.add_argument('--lm-loss', help='lm-loss', action='store_true')
    parser.add_argument('--lmk-loss', help='lmk-loss', action='store_true')
    parser.add_argument('--dynamic-upsampling', help='dynamic_upsampling', action='store_true')
    parser.add_argument( '--pretrained-model-name', help='pretrained_model_name', default=None, type=str)
    parser.add_argument('--init-translator', help='init_translator', action='store_true')
    parser.add_argument('--no-atten-mask', help='no_atten_mask', action='store_true')
    
  

    
    
    
    #optimization
    parser.add_argument('--max-update', help='max-update', default=None, type=int)
    parser.add_argument('--update-freq', help='update parameters every N_i batches, when in epoch i', default=None, type=int)
    
    #lr_schedule
    parser.add_argument('--warmup-updates', help='warmup_updates', default=None, type=int)
    
    #task  
    
    parser.add_argument('--data', help='dataset path', default=None, type=str)
    parser.add_argument('-t', '--task-name', help='task name', default=None, type=str)
    parser.add_argument('--pretrained-model-path', help='pretrained-model-path', default=None, type=str)
    parser.add_argument('--lmax-only-step', help='lmax_only_step', default=None, type=int)
    parser.add_argument('--debug', help='debug flag', action='store_true')
    parser.add_argument( '--pretrained-lm-name', help='pretrained_lm_name', default=None, type=str)
    parser.add_argument('--pretrained-lm-path', help='pretrained-lm-path', default=None, type=str)
    parser.add_argument('--eval-bleu-remove-bpe', help='eval_bleu_remove_bpe', default=None, type=str)
    
    
    
    args = parser.parse_args()    
    
    return args

def add_config(config, key1 , key2, value):
    config[key1] = {key2: value}
    
    
def set_config(config, key1 , key2, value):
    if type(value) == list:
        if len(value) >=2 :
            config[key1][key2] = value
            return
        temp =value[0]
        if temp is None:
            pass 
        else:
            config[key1][key2] = value  
    else:
        if value is None:
            pass 
        else:
            config[key1][key2] = value  
            
def del_config(config, key1 , key2):
    del config[key1][key2]             

         
# def del_config(config, key1 , key2):
#     del config[key1][key2]   

def main():
    
    args=parser_function()
    
    # base_architecture(args)
    if args.experiment is None:
        raise ValueError("Need input experiment id . Use [-e]")
    with open(args.basic_yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)    
    

    
#checkpoint
    key1='checkpoint'
    # set_config(config, key1 ,'save_dir', os.path.join(args.save_dir))
    set_config(config, key1 ,'save_dir', os.path.join(args.save_dir))
    set_config(config, key1 ,'save_interval_updates', args.save_interval_updates)
    
    
#common
    key1='common'
    set_config(config, key1 ,'wandb_entity', args.wandb_entity)
    set_config(config, key1 ,'wandb_project', args.wandb_project)
    set_config(config, key1 ,'fp16', args.fp16)
    
    
#criterion
    key1='criterion'
    set_config(config, key1 ,'_name', args.criterion_name) #nat_ctc_pred_rate_loss
#dataset
    key1='dataset'
    set_config(config, key1 ,'train_subset', args.train_subset) #nat_ctc_pred_rate_loss
    set_config(config, key1 ,'max_tokens', args.max_tokens) #nat_ctc_pred_rate_loss
    
#dataset_data_loading
    key1='dataset_data_loading'
#distributed_training
    key1='distributed_training'
    set_config(config, key1 ,'distributed_world_size', args.distributed_world_size) #num gpu
    
#lr_scheduler
#model
    key1='model'
    set_config(config, key1 ,'_name', args.model_name) # nonautoregressive_reorder_translation
    set_config(config, key1 ,'num_upsampling_rate', args.num_upsampling_rate) # nonautoregressive_reorder_translation
    set_config(config, key1 ,'lm_head_frozen', args.lm_head_frozen) 
    set_config(config, key1 ,'embedding_frozen', args.embedding_frozen) 
    set_config(config, key1 ,'upsample_fill_mask', args.upsample_fill_mask) 
    set_config(config, key1 ,'dynamic_rate', args.dynamic_rate) 
    set_config(config, key1 ,'lm_start_step', args.lm_start_step)
    set_config(config, key1 ,'voc_choosen', args.voc_choosen)    
    set_config(config, key1 ,'dropout', args.dropout)   
    set_config(config, key1 ,'has_eos', args.has_eos)  
    set_config(config, key1 ,'dynamic_upsampling', args.dynamic_upsampling)  
    set_config(config, key1 ,'lm_loss', args.lm_loss)  
    set_config(config, key1 ,'lmk_loss', args.lmk_loss)  
    set_config(config, key1 ,'pretrained_model_name', args.pretrained_model_name)
    set_config(config, key1 ,'pretrained_model_path', args.pretrained_model_path) 
    set_config(config, key1 ,'init_translator', args.init_translator) 
    set_config(config, key1 ,'no_atten_mask', args.no_atten_mask) 
    
    
    


    
    
#optimization
    key1='optimization'
    set_config(config, key1 ,'max_update', args.max_update)
    set_config(config, key1 ,'update_freq', [args.update_freq]) 
    set_config(config, key1 , 'stop_min_lr', 1e-09 )  
    
#optimizer
    key1='optimizer'
    if args.task_name != 'transaltion_ctcpmlm_rate' :
        del config[key1]['groups']
        set_config(config, key1 ,'_name', 'adam') 
        set_config(config, key1 ,'adam_betas', [0.9,0.98]) 
        set_config(config, key1 ,'adam_eps', 1e-08) 
        set_config(config, key1 ,'weight_decay', 0.01) 
     
        
#lr_scheduler 
    key1='lr_scheduler'
    if args.task_name != 'transaltion_ctcpmlm_rate' : 
        del config[key1]
        add_config(config, key1 , '_name', 'inverse_sqrt') 
        set_config(config, key1 , 'warmup_updates', args.warmup_updates) 
        set_config(config, key1 , 'warmup_init_lr', 1e-07 )
        set_config(config, key1 , 'lr', [0.0002] )
     
#task    
    key1='task'
    set_config(config, key1 ,'data', args.data) 
    set_config(config, key1 ,'_name', args.task_name) 
    set_config(config, key1 ,'lmax_only_step', args.lmax_only_step) 
    set_config(config, key1 ,'debug', args.debug) 
    set_config(config, key1 ,'pretrained_lm_name', args.pretrained_lm_name)
    set_config(config, key1 ,'pretrained_lm_path', args.pretrained_lm_path) 
    set_config(config, key1 ,'eval_bleu_remove_bpe', args.eval_bleu_remove_bpe) 
    set_config(config, key1 ,'pretrained_model_path', args.pretrained_model_path)     
    
    

    

    os.makedirs(os.path.join(args.save_dir),exist_ok=True)
    output_file_path=os.path.join(args.save_dir, args.experiment+".yaml")
    with open(output_file_path, 'w') as f:
        yaml.dump(config, f)




if __name__ == '__main__':
    main()